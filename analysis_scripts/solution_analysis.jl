# vm/qg/va violation rates and transformer counts: julia --project=. analysis_scripts/solution_analysis.jl <dataset_path> [report_path]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JSON
using HDF5
using Statistics
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

function _datapoint_violations(case_data; epsilon = EPSILON)
    return (
        vm = _vm_violations(case_data; epsilon = epsilon),
        qg = _qg_violations(case_data; epsilon = epsilon),
        va = _va_violations(case_data; epsilon = epsilon),
    )
end

"""
(violated, total) pairs, one per datapoint, to the fraction of datapoints
with any violation and the mean violated fraction. total == 0 counts as 0.0.
"""
function _summarize(pairs)
    any_violation = [violated > 0 for (violated, total) in pairs]
    fractions = [total > 0 ? violated / total : 0.0 for (violated, total) in pairs]
    return Dict(
        "fraction_datapoints_with_violation" => mean(any_violation),
        "avg_fraction_components_violated" => mean(fractions),
        "num_datapoints" => length(pairs),
    )
end

"""
Directory to read datapoints from: a `dataset.h5`, the directory holding it,
or a perturbation folder whose datapoints sit in `baseline_acpf/`.
"""
function _resolve_dataset_dir(dataset_path::AbstractString)
    isfile(dataset_path) && endswith(dataset_path, ".h5") && return dirname(abspath(dataset_path))
    if isempty(dataset_datapoints(dataset_path)) && isdir(joinpath(dataset_path, "baseline_acpf"))
        return joinpath(dataset_path, "baseline_acpf")
    end
    return dataset_path
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
vm/qg/va violations summarized over every datapoint of a dataset.
`dataset_path` goes through `_resolve_dataset_dir`.
"""
function analyze_dataset(dataset_path::AbstractString; epsilon = EPSILON)
    dataset_dir = _resolve_dataset_dir(dataset_path)
    datapoints = dataset_datapoints(dataset_dir)
    isempty(datapoints) && error("No datapoints (dataset.h5 or <n>.json) found in $dataset_path (or its baseline_acpf/ subdirectory)")

    vm_pairs = Tuple{Int,Int}[]
    qg_pairs = Tuple{Int,Int}[]
    va_pairs = Tuple{Int,Int}[]
    num_tap_changing, num_phase_shifting = nothing, nothing
    _foreach_datapoint(dataset_dir, datapoints) do case_data
        v = _datapoint_violations(case_data; epsilon = epsilon)
        push!(vm_pairs, v.vm)
        push!(qg_pairs, v.qg)
        push!(va_pairs, v.va)
        if num_tap_changing === nothing
            num_tap_changing, num_phase_shifting = _count_transformers(case_data)
        end
    end

    return Dict(
        "dataset_path" => abspath(dataset_dir),
        "num_datapoints" => length(datapoints),
        "num_tap_changing_transformers" => num_tap_changing,
        "num_phase_shifting_transformers" => num_phase_shifting,
        "vm_violations" => _summarize(vm_pairs),
        "qg_violations" => _summarize(qg_pairs),
        "va_violations" => _summarize(va_pairs),
    )
end

function _write_report(summary, path)
    open(path, "w") do io
        JSON.print(io, summary, 4)
    end
    return path
end

function _print_summary(summary)
    @printf("transformers: %d tap-changing, %d phase-shifting\n",
            summary["num_tap_changing_transformers"], summary["num_phase_shifting_transformers"])
    for (label, key) in [("vm", "vm_violations"), ("qg", "qg_violations"), ("va", "va_violations")]
        s = summary[key]
        @printf("%-3s: %6.2f%% of datapoints have >=1 violation, avg %6.2f%% of components violated (n=%d)\n",
                label, 100 * s["fraction_datapoints_with_violation"],
                100 * s["avg_fraction_components_violated"], s["num_datapoints"])
    end
end

function main()
    CASE_NAME = "case9241_pegase"
    PERT_NAME = "extreme_pert"
    dataset_path = "../../bus_swap_data/test_cases/data/$CASE_NAME/$PERT_NAME"

    summary = analyze_dataset(dataset_path)
    _print_summary(summary)
    # write the report next to the datapoints analyze_dataset found
    report_path = joinpath(summary["dataset_path"], "violation_summary.json")
    _write_report(summary, report_path)
    println("Saved violation summary to $report_path")
    return summary
end

