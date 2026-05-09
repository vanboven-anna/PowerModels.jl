#
# Per-component timing breakdown for the bus-type-switching solver.
#
# Question this script answers:
#   "Of the wall clock for each swap technique, how much is spent in the
#   donor-selection step vs. running Newton-Raphson?"
#
# Methodology:
#   1. For each case + xlsx sample, run NR to convergence and capture the
#      post-NR state (pf_data, mapping_dict, jacobian, bus_assignment,
#      b1_violations) -- this is exactly the input the dispatcher
#      `perform_bus_swaps!` would see at the first violation-triggering
#      point in the outer loop.
#   2. Time each component IN ISOLATION:
#        a. one cold-start NR call (`_compute_ac_pf`)
#        b. one `perform_bus_swaps!` call per swap_technique (each rep
#           starts from a fresh deepcopy of the pre-swap state, so
#           every measurement is on identical input).
#   3. Aggregate per-case medians and report:
#        - selection cost per call (microseconds)
#        - selection cost as a multiple of one NR call (the "tax" you pay)
#        - estimated total selection time per sample, using the
#          swap-iter counts observed in the existing benchmark.
#
# This script is read-only on the project source -- it just calls the
# existing public + internal functions with timing wrappers.
#
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using PowerModels
using XLSX
using DataFrames
using Printf
using Statistics

# ----- locate datasets (mirrors practice_sensitivity_score.jl) ---------------

const REPO_ROOT     = abspath(joinpath(@__DIR__, ".."))
const REPO_MATPOWER = joinpath(REPO_ROOT, "test", "data", "matpower")
const BSD_ROOT      = joinpath(REPO_ROOT, "bus_swap_data")
const HAS_BSD       = isdir(BSD_ROOT)
const HAS_CONFIG    = isfile(joinpath(REPO_ROOT, "config.jl"))
HAS_CONFIG && include(joinpath(REPO_ROOT, "config.jl"))

function _case_path(case_name)
    if HAS_BSD
        p = joinpath(BSD_ROOT, "test_cases", "network_info", case_name, "$(case_name).m")
        isfile(p) && return p
    end
    if HAS_CONFIG
        try
            p = joinpath(DATA_PATH, "test_cases", "network_info", case_name, "$(case_name).m")
            isfile(p) && return p
        catch end
    end
    p = joinpath(REPO_MATPOWER, "$(case_name).m")
    return isfile(p) ? p : nothing
end

function _line_limits_path(case_name)
    if HAS_BSD
        p = joinpath(BSD_ROOT, "test_cases", "network_info", case_name, "pg_line_limits.txt")
        isfile(p) && return p
    end
    if HAS_CONFIG
        try
            p = joinpath(DATA_PATH, "test_cases", "network_info", case_name, "pg_line_limits.txt")
            isfile(p) && return p
        catch end
    end
    return nothing
end

function _loads_xlsx_path(case_name)
    cands = String[]
    HAS_BSD && push!(cands, joinpath(BSD_ROOT, "test_cases", "data", case_name, "loads"))
    if HAS_CONFIG
        try push!(cands, joinpath(TESTCASE_PATH, "data", case_name, "loads")) catch end
    end
    for dir in cands
        isdir(dir) || continue
        files = filter(f -> endswith(f, ".xlsx"), readdir(dir))
        isempty(files) && continue
        return joinpath(dir, files[1])
    end
    return nothing
end

function _all_pairs_pv_pairs(data)
    gen_bus_ids = unique([gen["gen_bus"] for gen in values(data["gen"])])
    load_bus_ids = [bus["bus_i"] for bus in values(data["bus"])]
    return Dict{Int, Vector{Int}}(b => copy(gen_bus_ids) for b in load_bus_ids)
end

function _apply_xlsx_row!(data, row)
    for (i, l) in pairs(data["load"])
        l["pd"] = row["pd_$i"]
        l["qd"] = row["qd_$i"]
    end
    for (i, g) in pairs(data["gen"])
        g["pg"] = row["pg_$i"]
    end
end

function _prepare_test_case!(test_case, case_name)
    for gen in values(test_case["gen"])
        bus = test_case["bus"][string(gen["gen_bus"])]
        bus["vmax"] = max(get(gen, "vg", bus["vmax"]), bus["vmax"])
    end
    p = _line_limits_path(case_name)
    if p !== nothing
        for (i, lim) in enumerate(split(read(p, String), ' '))
            isempty(strip(lim)) && continue
            haskey(test_case["branch"], string(i)) || continue
            test_case["branch"][string(i)]["rate_a"] = parse(Int, lim)
        end
    end
end

# ----- per-sample setup: run NR and capture post-NR state -------------------

# Returns a NamedTuple with everything `perform_bus_swaps!` needs, or nothing
# if NR diverged or this sample produces no V violations on PQ buses (in
# which case there's no "swap call" to time).
function _post_nr_state(base_data, row)
    data = deepcopy(base_data)
    _apply_xlsx_row!(data, row)
    data["pv_pairs"] = _all_pairs_pv_pairs(data)

    pf_data = PowerModels.instantiate_pf_data(data)
    pf_data.data["pv_bus_inds"] = [i for (i, bt) in enumerate(pf_data.bus_type_idx) if bt == 2]
    pf_data.data["prev_swaps"]  = Dict(b => Int[] for b in 1:length(pf_data.bus_type_idx))

    mapping_dict, J0_map = PowerModels.map_types_to_variable_indices(pf_data)
    pf_result, jacobian, _ = try
        PowerModels._compute_ac_pf(pf_data, mapping_dict, J0_map; flat_start = true)
    catch
        return nothing
    end
    (pf_result.x_converged || pf_result.f_converged) || return nothing

    am = pf_data.am
    bus_assignment = Dict{String, Any}()
    for (i, bus) in pf_data.data["bus"]
        bus["bus_type"] == 4 && continue
        bus_idx = am.bus_to_idx[bus["index"]]
        bus_assignment[i] = Dict{String, Float64}(
            "vm"      => pf_data.vm_idx[bus_idx],
            "va"      => pf_data.va_idx[bus_idx],
            "bus_idx" => Float64(bus["index"]),
            "vmin"    => bus["vmin"],
            "vmax"    => bus["vmax"],
        )
    end

    b1 = Tuple{Int64, Float64, Float64}[]
    for (i, bt) in enumerate(pf_data.bus_type_idx)
        bt == 1 || continue
        bus = pf_data.data["bus"][string(am.idx_to_bus[i])]
        V = pf_data.vm_idx[i]
        if V < bus["vmin"]
            push!(b1, (i, bus["vmin"] - V, bus["vmin"]))
        elseif V > bus["vmax"]
            push!(b1, (i, bus["vmax"] - V, bus["vmax"]))
        end
    end
    isempty(b1) && return nothing

    return (
        pf_data        = pf_data,
        mapping_dict   = mapping_dict,
        J0_map         = J0_map,
        jacobian       = jacobian,
        bus_assignment = bus_assignment,
        b1_violations  = b1,
        n_pq           = count(==(1), pf_data.bus_type_idx),
        n_pv           = count(==(2), pf_data.bus_type_idx),
        n_buses        = length(pf_data.bus_type_idx),
    )
end

# ----- timing helpers --------------------------------------------------------

# Time the function `f` over `reps` runs after one warmup. Returns median.
# Setup that should NOT be measured goes inside `setup`, which is called
# fresh before every timed call (so each call sees identical input).
function _median_time(setup, f; reps = 10)
    s = setup(); f(s)               # warmup
    times = Vector{Float64}(undef, reps)
    for k in 1:reps
        s = setup()
        times[k] = @elapsed f(s)
    end
    return median(times)
end

# One cold-start NR call (`_compute_ac_pf` with flat_start=true).
function _time_nr_call(state; reps = 10)
    setup = function()
        pf_d = deepcopy(state.pf_data)
        md, jm = PowerModels.map_types_to_variable_indices(pf_d)
        return (pf_d, md, jm)
    end
    return _median_time(setup,
        s -> PowerModels._compute_ac_pf(s[1], s[2], s[3]; flat_start = true);
        reps = reps)
end

# One `perform_bus_swaps!` call for a chosen strategy.
function _time_swap_call(state, swap_technique, use_smw, score_collat; reps = 10)
    flags = PowerModels.SwapFlags(
        swap_technique         = swap_technique,
        obo                    = true,
        use_smw_warmstart      = use_smw,
        score_collateral_aware = score_collat,
    )
    setup = function()
        pf_d = deepcopy(state.pf_data)
        pf_d.data["pv_bus_inds"] = copy(state.pf_data.data["pv_bus_inds"])
        pf_d.data["prev_swaps"]  = Dict(b => copy(v) for (b, v) in state.pf_data.data["prev_swaps"])
        ba    = deepcopy(state.bus_assignment)
        viols = Dict("b1"  => deepcopy(state.b1_violations),
                     "b2"  => Vector{Any}(),
                     "b6v" => Vector{Any}(),
                     "b6q" => Vector{Any}())
        return (pf_d, copy(pf_d.bus_type_idx), Dict{Int64,Int64}(), ba, Ref(false), viols)
    end
    f = s -> PowerModels.perform_bus_swaps!(
        s[1], state.mapping_dict, s[2], s[3], state.jacobian[end],
        s[4], s[5], s[6], flags)
    return _median_time(setup, f; reps = reps)
end

# Wall clock of one full `compute_ac_pf_mult_buses` run on a sample.
function _time_full_solve(base_data, row, kwargs; reps = 3)
    setup = function()
        d = deepcopy(base_data)
        _apply_xlsx_row!(d, row)
        d["pv_pairs"] = _all_pairs_pv_pairs(d)
        return (d,)
    end
    f = s -> PowerModels.compute_ac_pf_mult_buses(s[1]; kwargs...)
    return _median_time(setup, f; reps = reps)
end

# ----- per-case profiler -----------------------------------------------------

const STRATS = [
    ("nearest_gen",   "nearest_gen",       false, false),
    ("qv_inv",        "qv_inv",            false, false),
    ("sens_score",    "sensitivity_score", false, false),
    ("sens+SMW",      "sensitivity_score", true,  false),
    ("sens+collat",   "sensitivity_score", false, true ),
]

# Compute feasibility metrics from a compute_ac_pf_mult_buses result.
function _feas_metrics(data, result)
    soln = get(result, "solution", nothing)
    if isnothing(soln) || !haskey(soln, "bus")
        return (feasible = false, nv_v = 0, nv_q = 0, mag_v = NaN, mag_q = NaN, converged = false)
    end
    converged = !any(b -> b["vm"] == -1.0, values(soln["bus"]))
    if !converged
        return (feasible = false, nv_v = 0, nv_q = 0, mag_v = NaN, mag_q = NaN, converged = false)
    end
    nv_v = 0; nv_q = 0; mag_v = 0.0; mag_q = 0.0
    for (s, b) in soln["bus"]
        bd = data["bus"][s]
        b["vm"] < bd["vmin"] - 1e-6 && (nv_v += 1; mag_v += bd["vmin"] - b["vm"])
        b["vm"] > bd["vmax"] + 1e-6 && (nv_v += 1; mag_v += b["vm"] - bd["vmax"])
    end
    for (s, g) in soln["gen"]
        gd = data["gen"][s]
        g["qg"] < gd["qmin"] - 1e-6 && (nv_q += 1; mag_q += gd["qmin"] - g["qg"])
        g["qg"] > gd["qmax"] + 1e-6 && (nv_q += 1; mag_q += g["qg"] - gd["qmax"])
    end
    return (feasible = (nv_v == 0 && nv_q == 0), nv_v = nv_v, nv_q = nv_q,
            mag_v = mag_v, mag_q = mag_q, converged = true)
end

function profile_case(case_name; max_samples = 25, reps = 10, max_acpf = 50)
    path = _case_path(case_name)
    path === nothing && (println("[skip] $case_name: no .m file"); return)
    xlsx = _loads_xlsx_path(case_name)
    xlsx === nothing && (println("[skip] $case_name: no xlsx loads file"); return)

    base = PowerModels.parse_file(path)
    _prepare_test_case!(base, case_name)
    df = DataFrame(XLSX.readtable(xlsx, "loads"))
    n = min(nrow(df), max_samples)

    println("=== $case_name  --  $(basename(xlsx)) ($(n)/$(nrow(df)) samples, max_acpf=$max_acpf) ===")

    nr_times    = Float64[]
    swap_times  = Dict(label => Float64[] for (label, _, _, _) in STRATS)
    full_times  = Dict(label => Float64[] for (label, _, _, _) in STRATS)
    swap_iters  = Dict(label => Int[]     for (label, _, _, _) in STRATS)
    jac_iters   = Dict(label => Int[]     for (label, _, _, _) in STRATS)
    feas_flag   = Dict(label => Bool[]    for (label, _, _, _) in STRATS)
    nv_v_arr    = Dict(label => Int[]     for (label, _, _, _) in STRATS)
    mag_v_arr   = Dict(label => Float64[] for (label, _, _, _) in STRATS)
    nv_q_arr    = Dict(label => Int[]     for (label, _, _, _) in STRATS)
    samples_used = 0
    n_buses, n_pv, n_pq = 0, 0, 0
    # Track baseline (no swap loop) feasibility too, for the % improvement column.
    base_mag_v = Float64[]
    base_feas  = Bool[]

    for s in 1:n
        row = df[s, :]
        st = _post_nr_state(base, row)
        st === nothing && continue
        samples_used += 1
        n_buses, n_pv, n_pq = st.n_buses, st.n_pv, st.n_pq

        push!(nr_times, _time_nr_call(st; reps = reps))
        for (label, tech, smw, col) in STRATS
            push!(swap_times[label], _time_swap_call(st, tech, smw, col; reps = reps))
        end

        # Baseline (no swap loop) on this sample, to anchor the |V|-reduction column.
        d_base = deepcopy(base); _apply_xlsx_row!(d_base, row)
        r_base = PowerModels.compute_ac_pf(d_base; mapping = true, enforce_q_lims = false)
        m_base = _feas_metrics(d_base, r_base)
        push!(base_mag_v, m_base.mag_v)
        push!(base_feas,  m_base.feasible)

        # Full-solve wall clock + iter counts + feasibility per strategy.
        for (label, tech, smw, col) in STRATS
            kwargs = (swap_technique = tech, obo = true,
                      use_smw_warmstart = smw, score_collateral_aware = col,
                      max_acpf = max_acpf)
            d = deepcopy(base)
            _apply_xlsx_row!(d, row)
            d["pv_pairs"] = _all_pairs_pv_pairs(d)
            t0 = time_ns()
            r = PowerModels.compute_ac_pf_mult_buses(d; kwargs...)
            push!(full_times[label], (time_ns() - t0) / 1e9)
            push!(swap_iters[label], length(get(r, "solution_history", [])))
            jhist = get(r, "jacobian_history", Any[])
            push!(jac_iters[label], count(j -> !(j isa AbstractVector && isempty(j)), jhist))
            m = _feas_metrics(d, r)
            push!(feas_flag[label], m.feasible)
            push!(nv_v_arr[label],  m.nv_v)
            push!(mag_v_arr[label], m.mag_v)
            push!(nv_q_arr[label],  m.nv_q)
        end
    end

    println("  n_buses=$n_buses  PQ=$n_pq  PV=$n_pv   samples_with_violations=$samples_used")
    if samples_used == 0
        println("  no samples had V violations on PQ buses; skipping breakdown")
        println()
        return
    end

    nr_med_us = median(nr_times) * 1e6
    @printf "\n  per-call cost (median over %d reps, %d samples):\n" reps samples_used
    @printf "  %-22s %10s   %10s\n" "phase" "µs/call" "× one NR"
    println("  " * "-"^48)
    @printf "  %-22s %10.1f   %10.2f\n" "_compute_ac_pf (NR)" nr_med_us 1.00
    for (label, _, _, _) in STRATS
        med_us = median(swap_times[label]) * 1e6
        @printf "  %-22s %10.1f   %10.3f\n" "swap! [$label]" med_us (med_us / nr_med_us)
    end

    @printf "\n  timing breakdown (median over %d samples):\n" samples_used
    @printf "  %-22s %5s %5s %10s %10s %12s %7s\n" "strategy" "swp" "jac" "T_swap(ms)" "T_NR(ms)" "T_full(ms)" "sel%"
    println("  " * "-"^78)
    for (label, _, _, _) in STRATS
        med_swap_us = median(swap_times[label]) * 1e6
        med_swp     = median(swap_iters[label])
        med_jac     = median(jac_iters[label])
        T_swap_ms = med_swap_us * med_swp / 1000
        T_full_ms = median(full_times[label]) * 1000
        T_NR_ms   = max(0.0, T_full_ms - T_swap_ms)
        sel_pct   = T_full_ms > 0 ? 100 * T_swap_ms / T_full_ms : 0.0
        @printf "  %-22s %5d %5d %10.3f %10.3f %12.3f %7.1f\n" label med_swp med_jac T_swap_ms T_NR_ms T_full_ms sel_pct
    end

    # Feasibility breakdown.
    base_feas_pct = 100 * mean(base_feas)
    base_meanV    = mean(filter(!isnan, base_mag_v))
    @printf "\n  feasibility breakdown (over %d samples):\n" samples_used
    @printf "  %-22s %7s %10s %7s %7s %12s\n" "strategy" "feas%" "mean|V|" "mean#V" "mean#Q" "ΔV vs base"
    println("  " * "-"^70)
    @printf "  %-22s %7.1f %10.4f %7s %7s %12s\n" "(baseline / no swap)" base_feas_pct base_meanV "—" "—" "—"
    for (label, _, _, _) in STRATS
        feas_pct = 100 * mean(feas_flag[label])
        mean_mv  = mean(filter(!isnan, mag_v_arr[label]))
        mean_nv  = mean(nv_v_arr[label])
        mean_nq  = mean(nv_q_arr[label])
        # Per-sample paired ΔV: compare each strategy result to its own
        # baseline result on the same sample, then average. Counts an
        # improvement (>0%) where the strategy reduces the |V| residual.
        deltas = Float64[]
        for k in eachindex(base_mag_v)
            isnan(base_mag_v[k]) && continue
            isnan(mag_v_arr[label][k]) && continue
            base_mag_v[k] <= 1e-9 && continue   # baseline already feasible
            push!(deltas, 100 * (1 - mag_v_arr[label][k] / base_mag_v[k]))
        end
        red_pct_str = isempty(deltas) ? "—" : @sprintf("%+.1f%%", mean(deltas))
        @printf "  %-22s %7.1f %10.4f %7.2f %7.2f %12s\n" label feas_pct mean_mv mean_nv mean_nq red_pct_str
    end
    println()
end

# ----- main ------------------------------------------------------------------

PowerModels.silence()
PowerModels.logger_config!("warn")

for case in ["case14", "case57", "case300"]
    profile_case(case;
        max_samples = case == "case300" ? 10 : 25,
        reps        = 10,
        max_acpf    = 50,   # matches SwapFlags default
    )
end

println("Notes:")
println("  • '× one NR' is the swap-call cost normalized to one full NR call.")
println("    Values < 1 mean donor selection is cheaper than a Newton-Raphson solve.")
println("  • 'jac' counts the total Newton iterations a sample uses across its")
println("    swap loop; 'swp' counts the outer-loop bus-type-switching iterations.")
println("  • 'T_swap' = swp × (median µs/swap-call). 'T_NR_est' is the residual")
println("    after subtracting T_swap from the actually observed full-solve")
println("    wall clock -- so it includes NR plus warm-start bookkeeping.")
