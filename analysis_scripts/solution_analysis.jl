# vm/qg/va violation rates, solve times and transformer counts: julia --project=. analysis_scripts/solution_analysis.jl <dataset_path> [report_path]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JSON
using HDF5
using Statistics
using PowerModels
# dataset_datapoints / load_datapoint
isdefined(@__MODULE__, :load_datapoint) ||
    include(joinpath(@__DIR__, "..", "run_scripts", "unpack_data.jl"))
using Printf

const EPSILON = 1e-5

"""
Bus vm violations, PQ buses (bus_type == 1) only: PV/slack vm is pinned to a
generator's vg, and bus_type 4 is out of service.
"""
function _vm_violations(case_data; epsilon = EPSILON)
    violated, total = 0, 0
    for bus in values(case_data["bus"])
        if bus["bus_type"] != 1
            continue
        end
        total += 1
        vm = bus["vm"]
        if vm > bus["vmax"] + epsilon || vm < bus["vmin"] - epsilon
            violated += 1
        end
    end
    return violated, total
end

"Generator qg violations, restricted to in-service generators."
function _qg_violations(case_data; epsilon = EPSILON)
    violated, total = 0, 0
    for gen in values(case_data["gen"])
        if gen["gen_status"] == 0
            continue
        end
        total += 1
        qg = gen["qg"]
        if qg > gen["qmax"] + epsilon || qg < gen["qmin"] - epsilon
            violated += 1
        end
    end
    return violated, total
end

"""
Branch voltage-angle-difference violations, restricted to in-service
branches: va_f - va_t checked against [angmin, angmax].
"""
function _va_violations(case_data; epsilon = EPSILON)
    violated, total = 0, 0
    for branch in values(case_data["branch"])
        if branch["br_status"] == 0
            continue
        end
        f_bus = case_data["bus"][string(branch["f_bus"])]
        t_bus = case_data["bus"][string(branch["t_bus"])]
        total += 1
        vad = f_bus["va"] - t_bus["va"]
        if vad > branch["angmax"] + epsilon || vad < branch["angmin"] - epsilon
            violated += 1
        end
    end
    return violated, total
end

"""
Transformer branches that are tap-changing (`tap != 1.0`) vs phase-shifting
(`shift` not 0 or a full rotation), as `classify_transformers` splits them.
Not mutually exclusive.
"""
function _count_transformers(case_data; angle_eps = 1e-6)
    full_rotation = 2 * pi  # shift is in radians
    tap_changing, phase_shifting = 0, 0
    for branch in values(case_data["branch"])
        get(branch, "transformer", false) || continue
        if branch["tap"] != 1.0
            tap_changing += 1
        end
        shift_mod = mod(branch["shift"], full_rotation)
        is_no_shift = isapprox(shift_mod, 0.0; atol = angle_eps) ||
                      isapprox(shift_mod, full_rotation; atol = angle_eps)
        if !is_no_shift
            phase_shifting += 1
        end
    end
    return tap_changing, phase_shifting
end

"""
Branch thermal violations, in-service branches only: max(|S_from|, |S_to|)
against `rate_a`. Branches without a rate_a are unlimited, as in the models.
"""
function _sm_violations(case_data; epsilon = EPSILON)
    violated, total = 0, 0
    flows = calc_branch_flow_ac(case_data)
    for (i, branch) in case_data["branch"]
        if branch["br_status"] == 0 || !haskey(branch, "rate_a")
            continue
        end
        total += 1
        f = flows["branch"][i]
        sm = max(hypot(f["pf"], f["qf"]), hypot(f["pt"], f["qt"]))
        if sm > branch["rate_a"] + epsilon
            violated += 1
        end
    end
    return violated, total
end

function _datapoint_violations(case_data; epsilon = EPSILON)
    return (
        vm = _vm_violations(case_data; epsilon = epsilon),
        qg = _qg_violations(case_data; epsilon = epsilon),
        va = _va_violations(case_data; epsilon = epsilon),
        sm = _sm_violations(case_data; epsilon = epsilon),
    )
end

"mean and the 25th/50th/75th percentiles of `values`; empty -> nothing."
function _stats(values)
    isempty(values) && return nothing
    v = collect(float.(values))
    return Dict(
        "mean" => mean(v),
        "p25" => quantile(v, 0.25),
        "p50" => quantile(v, 0.50),
        "p75" => quantile(v, 0.75),
        "min" => minimum(v),
        "max" => maximum(v),
        "num_datapoints" => length(v),
    )
end

"""
(violated, total) pairs, one per datapoint, to the fraction of datapoints
with any violation and the violated-fraction stats. total == 0 counts as 0.0.
"""
function _summarize(pairs)
    any_violation = [violated > 0 for (violated, total) in pairs]
    fractions = [total > 0 ? violated / total : 0.0 for (violated, total) in pairs]
    q = _stats(fractions)
    return Dict(
        "fraction_datapoints_with_violation" => isempty(pairs) ? 0.0 : mean(any_violation),
        "avg_fraction_components_violated" => q === nothing ? 0.0 : q["mean"],
        "fraction_components_violated_quartiles" => q,
        "num_datapoints" => length(pairs),
    )
end

"""
Directory to read datapoints from: a `dataset.h5`, the directory holding it,
or a perturbation folder whose datapoints sit in `<dat_type>/`.
"""
function _resolve_dataset_dir(dataset_path::AbstractString, dat_type::Union{Nothing,AbstractString} = nothing)
    isfile(dataset_path) && endswith(dataset_path, ".h5") && return dirname(abspath(dataset_path))
    if dat_type !== nothing && isempty(dataset_datapoints(dataset_path)) &&
       isdir(joinpath(dataset_path, dat_type))
        return joinpath(dataset_path, dat_type)
    end
    return dataset_path
end

"""
Solve-time stats from the archive's `acpf_time` / `dcopf_time` columns,
restricted to `datapoints`. Empty Dict when the archive recorded none.
"""
function _timing_stats(dir, datapoints)
    has_dataset_h5(dir) || return Dict{String,Any}()
    timing = read_timing(dataset_h5_path(dir))
    haskey(timing, "datapoint") || return Dict{String,Any}()
    rows = [findfirst(==(dp), timing["datapoint"]) for dp in datapoints]
    rows = [r for r in rows if r !== nothing]
    out = Dict{String,Any}()
    for f in ("acpf_time", "dcopf_time")
        haskey(timing, f) || continue
        vals = Float64.(timing[f][rows])
        s = _stats(vals)
        s === nothing || (out[f] = s)
    end
    if haskey(timing, "acpf_time") && haskey(timing, "dcopf_time") && !isempty(rows)
        total = Float64.(timing["acpf_time"][rows]) .+ Float64.(timing["dcopf_time"][rows])
        out["total_time"] = _stats(total)
    end
    return out
end

"""
`f(case_data)` over `datapoints`, one case alive at a time. Opens the archive
once rather than per datapoint as `load_datapoint` would.
"""
function _foreach_datapoint(f, dir, datapoints)
    if has_dataset_h5(dir)
        h5open(dataset_h5_path(dir), "r") do fid
            stored = read(fid["datapoint"])
            for dp in datapoints
                row = findfirst(==(dp), stored)
                row === nothing && error("analyze_dataset: datapoint $dp not in $(dataset_h5_path(dir))")
                f(_build_sample(fid, row))
            end
        end
    else
        for dp in datapoints
            f(load_datapoint(dir, dp))
        end
    end
    return nothing
end

"""
vm/qg/va violations and solve times summarized over every datapoint of a
dataset. `dataset_path`/`dat_type` go through `_resolve_dataset_dir`.
"""
function analyze_dataset(dataset_path::AbstractString,
                         dat_type::Union{Nothing,AbstractString} = nothing;
                         epsilon = EPSILON)
    dataset_dir = _resolve_dataset_dir(dataset_path, dat_type)
    datapoints = dataset_datapoints(dataset_dir)
    isempty(datapoints) && error("No datapoints (dataset.h5 or <n>.json) found in $dataset_path (or its $(dat_type === nothing ? "<dat_type>" : dat_type)/ subdirectory)")

    vm_pairs = Tuple{Int,Int}[]
    qg_pairs = Tuple{Int,Int}[]
    va_pairs = Tuple{Int,Int}[]
    sm_pairs = Tuple{Int,Int}[]
    num_tap_changing, num_phase_shifting = nothing, nothing
    _foreach_datapoint(dataset_dir, datapoints) do case_data
        v = _datapoint_violations(case_data; epsilon = epsilon)
        push!(vm_pairs, v.vm)
        push!(qg_pairs, v.qg)
        push!(va_pairs, v.va)
        push!(sm_pairs, v.sm)
        if num_tap_changing === nothing
            num_tap_changing, num_phase_shifting = _count_transformers(case_data)
        end
    end

    return Dict(
        "dataset_path" => abspath(dataset_dir),
        "dat_type" => dat_type === nothing ? basename(abspath(dataset_dir)) : dat_type,
        "num_datapoints" => length(datapoints),
        "solve_times" => _timing_stats(dataset_dir, datapoints),
        "num_tap_changing_transformers" => num_tap_changing,
        "num_phase_shifting_transformers" => num_phase_shifting,
        "vm_violations" => _summarize(vm_pairs),
        "qg_violations" => _summarize(qg_pairs),
        "va_violations" => _summarize(va_pairs),
        "sm_violations" => _summarize(sm_pairs),
    )
end

function _write_report(summary, path)
    open(path, "w") do io
        JSON.print(io, summary, 4)
    end
    return path
end

function _print_summary(summary)
    @printf("dat_type: %s (n=%d)\n", summary["dat_type"], summary["num_datapoints"])
    @printf("transformers: %d tap-changing, %d phase-shifting\n",
            summary["num_tap_changing_transformers"], summary["num_phase_shifting_transformers"])
    for (label, key) in [("vm", "vm_violations"), ("qg", "qg_violations"),
                        ("va", "va_violations"), ("sm", "sm_violations")]
        s = summary[key]
        @printf("%-3s: %6.2f%% of datapoints have >=1 violation, avg %6.2f%% of components violated (n=%d)\n",
                label, 100 * s["fraction_datapoints_with_violation"],
                100 * s["avg_fraction_components_violated"], s["num_datapoints"])
        q = s["fraction_components_violated_quartiles"]
        q === nothing || @printf("     quartiles of %% violated: p25 %6.2f%%, p50 %6.2f%%, p75 %6.2f%%\n",
                                 100 * q["p25"], 100 * q["p50"], 100 * q["p75"])
    end
    times = summary["solve_times"]
    if isempty(times)
        println("solve times: none recorded")
    else
        for key in ("acpf_time", "dcopf_time", "total_time")
            haskey(times, key) || continue
            t = times[key]
            @printf("%-10s: avg %8.4f s, p25 %8.4f, p50 %8.4f, p75 %8.4f\n",
                    key, t["mean"], t["p25"], t["p50"], t["p75"])
        end
    end
end

function main()
    for CASE_NAME in ["case14", "case57", "case118", "case300", "case2869_pegase", "case7336"]
        PERT_NAME = "extreme_pert"
        DAT_TYPE = "dc_ac_pf"
        dataset_path = "../../bus_swap_data/test_cases/data/$CASE_NAME/$PERT_NAME"

        summary = analyze_dataset(dataset_path, DAT_TYPE)
        _print_summary(summary)
        # write the report next to the datapoints analyze_dataset found
        report_path = joinpath(summary["dataset_path"], "violation_summary.json")
        _write_report(summary, report_path)
        println("Saved violation summary to $report_path")
    end
    # CASE_NAME = "case14"

    # return summary
end

