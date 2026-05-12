#
# Diagnostic: case300 sensitivity_score / SMW.
#
# Investigates two questions raised when reviewing the main benchmark numbers:
#
# (1) Reporting vs. optimization metric. The algorithm in pf.jl minimizes
#     violation_mag = sum(|V| overruns) + sum(|Q| overruns). The main
#     practice script reports only |V|. On case300, sens+SMW showed a much
#     larger |V| than baseline (0.47 vs 0.10), which looked like a regression
#     but might just be the algorithm trading Q magnitude for V magnitude.
#     This file reports both |V| and |Q| (plus their sum) and lists per-sample
#     "worse than baseline by total mag" counts -- if best-solution tracking
#     is correct, that count must be zero.
#
# (2) Constraint ordering. perform_bus_swaps! handles violations as
#     Q-batch -> b6v -> b1 (PV->PQ). When obo=false the b1 batch can apply
#     many PV->PQ swaps in one iter, sometimes generating new V violations.
#     This file exercises a new flag `b1_obo` (added to SwapFlags) that caps
#     only the b1 batch to one swap per iter, leaving the Q / b6v batches
#     uncapped.
#
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using PowerModels
using XLSX
using DataFrames
using Printf
using Random
using Statistics

Random.seed!(2026)

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const BSD_ROOT  = joinpath(REPO_ROOT, "bus_swap_data")
const CASE_NAME = "case300"
const MAX_SAMPLES = 15

# ----- case prep (mirrors practice_sensitivity_score.jl) ---------------------

function _case_path()
    return joinpath(BSD_ROOT, "test_cases", "network_info", CASE_NAME, "$(CASE_NAME).m")
end

function _loads_xlsx_path()
    dir = joinpath(BSD_ROOT, "test_cases", "data", CASE_NAME, "loads")
    files = filter(f -> endswith(f, ".xlsx"), readdir(dir))
    return joinpath(dir, files[1])
end

function _prepare_test_case!(test_case)
    for gen in values(test_case["gen"])
        gen_bus = gen["gen_bus"]
        bus = test_case["bus"][string(gen_bus)]
        bus["vmax"] = max(get(gen, "vg", bus["vmax"]), bus["vmax"])
    end
    line_lim_pth = joinpath(BSD_ROOT, "test_cases", "network_info", CASE_NAME, "pg_line_limits.txt")
    if isfile(line_lim_pth)
        line_limits = split(read(line_lim_pth, String), ' ')
        for (i, lim) in enumerate(line_limits)
            isempty(strip(lim)) && continue
            haskey(test_case["branch"], string(i)) || continue
            test_case["branch"][string(i)]["rate_a"] = parse(Int, lim)
        end
    end
end

function _apply_xlsx_row!(data, row)
    for (load_ind, load) in pairs(data["load"])
        load["pd"] = row["pd_$load_ind"]
        load["qd"] = row["qd_$load_ind"]
    end
    for (gen_ind, gen) in pairs(data["gen"])
        gen["pg"] = row["pg_$gen_ind"]
    end
end

function _all_pv_pairs(data)
    gen_bus_ids = unique([gen["gen_bus"] for gen in values(data["gen"])])
    load_bus_ids = [bus["bus_i"] for bus in values(data["bus"])]
    return Dict{Int, Vector{Int}}(b => copy(gen_bus_ids) for b in load_bus_ids)
end

# ----- detailed violation accounting (V and Q separately) --------------------

function _violations(data, result)
    nv_v = 0; nv_q = 0
    mag_v = 0.0; mag_q = 0.0
    soln = get(result, "solution", nothing)
    (isnothing(soln) || !haskey(soln, "bus")) && return (nv_v, nv_q, mag_v, mag_q, false)
    converged = !any(b -> b["vm"] == -1.0, values(soln["bus"]))
    !converged && return (nv_v, nv_q, mag_v, mag_q, false)
    for (s, b) in soln["bus"]
        bd = data["bus"][s]
        if b["vm"] < bd["vmin"] - 1e-6; nv_v += 1; mag_v += bd["vmin"] - b["vm"]; end
        if b["vm"] > bd["vmax"] + 1e-6; nv_v += 1; mag_v += b["vm"] - bd["vmax"]; end
    end
    for (s, g) in soln["gen"]
        gd = data["gen"][s]
        if g["qg"] < gd["qmin"] - 1e-6; nv_q += 1; mag_q += gd["qmin"] - g["qg"]; end
        if g["qg"] > gd["qmax"] + 1e-6; nv_q += 1; mag_q += g["qg"] - gd["qmax"]; end
    end
    return (nv_v, nv_q, mag_v, mag_q, true)
end

function _jac_iters(result)
    haskey(result, "jacobian_history") || return 0
    return count(j -> !(j isa AbstractVector && isempty(j)), result["jacobian_history"])
end

function _run_strategy(kind, kwargs, data)
    sample = deepcopy(data)
    t0 = time()
    result = if kind == :baseline
        PowerModels.compute_ac_pf(sample; mapping = true, enforce_q_lims = false)
    else
        PowerModels.compute_ac_pf_mult_buses(sample; kwargs...)
    end
    t = time() - t0
    nv_v, nv_q, mag_v, mag_q, converged = _violations(sample, result)
    swap_iters = (kind == :switch && haskey(result, "solution_history")) ? length(result["solution_history"]) : 1
    return (
        time = t, swap_iters = swap_iters, jac_iters = _jac_iters(result),
        nv_v = nv_v, nv_q = nv_q, mag_v = mag_v, mag_q = mag_q,
        mag_total = mag_v + mag_q,
        converged = converged, feasible = converged && nv_v == 0 && nv_q == 0,
    )
end

# ----- strategies under comparison ------------------------------------------

# Two axes: swap cap (obo / b1_obo / uncapped) x SMW warm-start (on/off).
const STRATEGIES = [
    ("baseline",                       :baseline, NamedTuple()),
    ("sens obo=true",                  :switch,   (obo = true,  swap_technique = "sensitivity_score")),
    ("sens obo=true +SMW",             :switch,   (obo = true,  swap_technique = "sensitivity_score", use_smw_warmstart = true)),
    ("sens obo=false",                 :switch,   (obo = false, swap_technique = "sensitivity_score")),
    ("sens obo=false +SMW",            :switch,   (obo = false, swap_technique = "sensitivity_score", use_smw_warmstart = true)),
    ("sens b1_obo only",               :switch,   (obo = false, b1_obo = true, swap_technique = "sensitivity_score")),
    ("sens b1_obo only +SMW",          :switch,   (obo = false, b1_obo = true, swap_technique = "sensitivity_score", use_smw_warmstart = true)),
]

# ----- driver ---------------------------------------------------------------

function main()
    PowerModels.logger_config!("warn")
    PowerModels.silence()

    case_path = _case_path()
    base_data = PowerModels.parse_file(case_path)
    _prepare_test_case!(base_data)
    pv_pairs = _all_pv_pairs(base_data)

    xlsx_path = _loads_xlsx_path()
    df = DataFrame(XLSX.readtable(xlsx_path, "loads"))
    n_samples = min(nrow(df), MAX_SAMPLES)

    println("=== $CASE_NAME diagnostic  --  $(basename(xlsx_path)) ($(n_samples)/$(nrow(df)) samples) ===")

    # JIT warmup so the first sample's timing isn't compile-dominated.
    print("  warmup...")
    let warm = deepcopy(base_data)
        warm["pv_pairs"] = deepcopy(pv_pairs)
        for (_, kind, kwargs) in STRATEGIES
            try; _run_strategy(kind, kwargs, warm); catch; end
        end
    end
    println(" done")

    results = Dict{String, Vector{NamedTuple}}(label => NamedTuple[] for (label, _, _) in STRATEGIES)
    for i in 1:n_samples
        sample_data = deepcopy(base_data)
        _apply_xlsx_row!(sample_data, df[i, :])
        sample_data["pv_pairs"] = deepcopy(pv_pairs)
        for (label, kind, kwargs) in STRATEGIES
            r = _run_strategy(kind, kwargs, sample_data)
            push!(results[label], r)
        end
    end

    # ---- Summary table with |V|, |Q|, and total ----
    @printf "\n  %-25s %6s %6s %8s %6s %8s %8s %7s %9s\n" "strategy" "feas%" "#V" "|V|" "#Q" "|Q|" "|V|+|Q|" "jac" "time(s)"
    println("  " * "-"^(25 + 6 + 6 + 8 + 6 + 8 + 8 + 7 + 9 + 8))
    for (label, _, _) in STRATEGIES
        rs = results[label]
        feas = 100 * mean(r.feasible for r in rs)
        nvv  = mean(r.nv_v for r in rs)
        magv = mean(r.mag_v for r in rs)
        nvq  = mean(r.nv_q for r in rs)
        magq = mean(r.mag_q for r in rs)
        magt = mean(r.mag_total for r in rs)
        jit  = mean(r.jac_iters for r in rs)
        tavg = mean(r.time for r in rs)
        @printf "  %-25s %6.1f %6.2f %8.4f %6.2f %8.4f %8.4f %7.2f %9.5f\n" label feas nvv magv nvq magq magt jit tavg
    end

    # ---- Paired W/T/L vs baseline by TOTAL mag (= the algorithm's objective) ----
    rs_base = results["baseline"]
    println("\n  Paired W/T/L vs baseline by |V|+|Q| (algorithm's objective):")
    for (label, _, _) in STRATEGIES
        label == "baseline" && continue
        rs = results[label]
        wins = 0; ties = 0; losses = 0
        for k in 1:length(rs)
            if     rs[k].mag_total < rs_base[k].mag_total - 1e-9; wins   += 1
            elseif rs[k].mag_total > rs_base[k].mag_total + 1e-9; losses += 1
            else;                                                  ties  += 1
            end
        end
        @printf "    %-25s W/T/L = %d/%d/%d  (mean %.4f -> %.4f)\n" label wins ties losses mean(r.mag_total for r in rs_base) mean(r.mag_total for r in rs)
    end

    # ---- Regressions: per-sample list of samples where strategy did worse than baseline by total mag ----
    # If best-solution tracking works, this must be zero on every row. Any nonzero
    # entry is a real bug in the swap-loop's best tracking (not a metric artifact).
    println("\n  Samples where strategy is WORSE than baseline by |V|+|Q| (should be 0 if best-tracking is correct):")
    for (label, _, _) in STRATEGIES
        label == "baseline" && continue
        rs = results[label]
        worse = [k for k in 1:length(rs) if rs[k].mag_total > rs_base[k].mag_total + 1e-9]
        if isempty(worse)
            @printf "    %-25s 0 regressions\n" label
        else
            # Show the first few offending samples with their mag values.
            shown = first(worse, 5)
            diffs = ["#$(k): $(round(rs_base[k].mag_total, digits=4))->$(round(rs[k].mag_total, digits=4))" for k in shown]
            extra = length(worse) > length(shown) ? " (...$(length(worse) - length(shown)) more)" : ""
            @printf "    %-25s %d regressions: %s%s\n" label length(worse) join(diffs, ", ") extra
        end
    end

    # ---- Quick view: |V| vs |Q| trade for sens+SMW ----
    # Re-stating the case300 question in absolute numbers: does sens+SMW lower |Q|
    # enough to justify its higher |V|, relative to baseline?
    println("\n  Per-strategy mean |V| / |Q| split (so you can see the trade):")
    for (label, _, _) in STRATEGIES
        rs = results[label]
        @printf "    %-25s |V|=%.4f  |Q|=%.4f  total=%.4f\n" label mean(r.mag_v for r in rs) mean(r.mag_q for r in rs) mean(r.mag_total for r in rs)
    end
    println()
end

main()
