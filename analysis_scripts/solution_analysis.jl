# Reads a directory of per-datapoint PowerModels case JSONs (as written by
# run_scripts/generate_dataset.jl's generate_data!/store_datapoint!, each file
# holding a perturbed case with the solved AC-PF setpoints baked into
# "bus"/"gen") and reports:
#   - the number of tap-changing and phase-shifting transformers in the
#     network (a fixed topology property, counted once off the first
#     datapoint - perturbations only change tap/shift values, not which
#     branches are transformers)
#   - for bus vm violations, generator qg violations, and branch
#     voltage-angle-difference (va) violations:
#       - the fraction of datapoints with at least one violation
#       - the average, across datapoints, of the fraction of components violated
#
# PV/slack buses (bus_type != 1) are excluded from vm violation checks: their
# vm is pinned to a generator's voltage setpoint (vg), so a high vm there
# reflects that setpoint rather than an actual voltage violation.
#
# Run from the repo root:
#   julia --project=. --startup-file=no analysis_scripts/solution_analysis.jl <dataset_path> [report_path]

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using JSON
using Statistics
using Printf

const EPSILON = 1e-5

"""
Bus vm violations, restricted to PQ buses (bus_type == 1). PV/slack buses
(bus_type 2/3) have vm pinned to a generator's vg setpoint -- a vm above
vmax there reflects that setpoint, not a real voltage violation -- and
bus_type 4 buses are out of service, so both are excluded.
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
Count the branches of `case_data` that Matpower encodes as transformers
(`transformer == true`) that are tap-changing vs. phase-shifting, mirroring
`run_scripts/generate_dataset.jl`'s `classify_transformers`: a branch is a
tap-changer if `tap != 1.0`, and a phase-shifter if `shift` differs from "no
shift" (0, or an equivalent full rotation of +-360 degrees / +-2*pi radians,
since some case files encode "no shift" that way). A transformer can be both,
so the two counts are not mutually exclusive.
"""
function _count_transformers(case_data; angle_eps = 1e-6)
    full_rotation = 2 * pi  # shift is stored in radians; 360 degrees == 2*pi
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
Aggregate a vector of (violated, total) pairs -- one per datapoint -- into
the fraction of datapoints with any violation and the average, across
datapoints, of the per-datapoint violated fraction. A datapoint with
total == 0 (no eligible components) contributes a violated fraction of 0.0
and counts as having no violation.
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
Read every `<datapoint>.json` case file in `dataset_path` and summarize
vm/qg/va violations across the dataset. Only files named as a bare integer
(the `store_datapoint!` naming convention) are treated as case files, so a
`violation_summary.json` report left over from a previous run in the same
directory is not picked back up and parsed as a case.
"""
function analyze_dataset(dataset_path::AbstractString; epsilon = EPSILON)
    files = sort(filter(f -> occursin(r"^\d+\.json$", f), readdir(dataset_path)))
    isempty(files) && error("No datapoint .json files found in $dataset_path")

    vm_pairs = Tuple{Int,Int}[]
    qg_pairs = Tuple{Int,Int}[]
    va_pairs = Tuple{Int,Int}[]
    num_tap_changing, num_phase_shifting = nothing, nothing
    for file in files
        case_data = JSON.parsefile(joinpath(dataset_path, file); dicttype = Dict{String,Any})
        v = _datapoint_violations(case_data; epsilon = epsilon)
        push!(vm_pairs, v.vm)
        push!(qg_pairs, v.qg)
        push!(va_pairs, v.va)
        if num_tap_changing === nothing
            # transformer topology is fixed by the network (perturbations only
            # change tap/shift values, not which branches are transformers),
            # so counting once off the first datapoint is enough
            num_tap_changing, num_phase_shifting = _count_transformers(case_data)
        end
    end

    return Dict(
        "dataset_path" => abspath(dataset_path),
        "num_files" => length(files),
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
    CASE_NAME = "case2869_pegase"
    PERT_NAME = "default_pert"
    dataset_path = "../../bus_swap_data/test_cases/data/$CASE_NAME/$PERT_NAME"
    report_path =  joinpath(dataset_path, "violation_summary.json")

    summary = analyze_dataset(dataset_path)
    _print_summary(summary)
    _write_report(summary, report_path)
    println("Saved violation summary to $report_path")
    return summary
end

