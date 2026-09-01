# For every test case in `bus_swap_data/test_cases/network_info`, generates a
# 10-datapoint dataset (`generate_data`) under each perturbation config in
# `perturbation_configs/`, then runs `solution_analysis.analyze_dataset` on
# the resulting dataset and writes its `violation_summary.json` report.
#
# A test case or (case, config) pair that fails to load/generate/analyze is
# logged and skipped rather than aborting the whole run.
#
# Run from the repo root:
#   julia --project=. --startup-file=no run_scripts/generate_and_analyze_all.jl

include("./generate_dataset.jl")
include("../analysis_scripts/solution_analysis.jl")

const NETWORK_INFO_DIR = joinpath(DATA_PATH, "test_cases/network_info")
const NUM_POINTS = 10

"""
List every case name in `network_info_dir` that has a `<case_name>.m`
Matpower file directly inside its like-named subdirectory (skips non-case
entries such as `.DS_Store` or an empty placeholder folder).
"""
function discover_case_names(network_info_dir = NETWORK_INFO_DIR)
    case_names = String[]
    for entry in sort(readdir(network_info_dir))
        case_dir = joinpath(network_info_dir, entry)
        if isdir(case_dir) && isfile(joinpath(case_dir, "$entry.m"))
            push!(case_names, entry)
        end
    end
    return case_names
end

"List the perturbation config filenames (e.g. `default_pert.json`) in `PERTURBATION_CONFIGS_DIR`."
function discover_pert_configs()
    return sort(filter(f -> endswith(f, ".json"), readdir(PERTURBATION_CONFIGS_DIR)))
end

"""
Run the full case x config sweep, optionally starting partway through the
(sorted) case list instead of from the beginning -- e.g. to resume after an
earlier run stalled on a particular case. `start_case`, if given, must be one
of the names `discover_case_names` finds; every case sorted before it is
skipped entirely (not even attempted).

`quiet` (default `true`) suppresses PowerModels' own Info/Warn console
output (e.g. "removing N cost terms", "the voltage setpoint on generator
... does not match ...") for the whole run, leaving only Error-level
PowerModels output and this script's own `println` progress messages. Pass
`quiet = false` to see Info/Warn messages again. `generate_data` resets
PowerModels' logger level itself on every call (so setting it once up front
wouldn't stick), so the level is set both here -- covering `parse_file`/
`prepare_test_case`, which run before the first `generate_data` call for
each case -- and passed through to every `generate_data` call via
`log_level`.

`skip_existing` (default `false`), when `true`, still iterates every case x
config combination but skips a combination entirely (no `generate_data`, no
`analyze_dataset`) if its output folder (`data/<case_name>/<pert_name>/`)
already exists -- so you can always run the full sweep from the start and it
naturally resumes past whatever's already been generated, without needing to
figure out a `start_case` by hand. Note this is a plain existence check, not
a completeness check: a folder left behind by a run that crashed partway
through (e.g. missing `violation_summary.json`, or fewer than `NUM_POINTS`
datapoints) still counts as "existing" and gets skipped.
"""
function main(; start_case::Union{Nothing,AbstractString} = nothing,
                quiet::Bool = true,
                config_lst = nothing,
                skip_existing::Bool = true)
    log_level = quiet ? "error" : "warn"
    PowerModels.logger_config!(log_level)

    case_names = discover_case_names()
    if start_case !== nothing
        start_idx = findfirst(==(start_case), case_names)
        start_idx === nothing && error("start_case = \"$start_case\" not found among discovered case names: $case_names")
        case_names = case_names[start_idx:end]
    end
    pert_configs = isnothing(config_lst) ? discover_pert_configs() : config_lst
    println("Found $(length(case_names)) test cases and $(length(pert_configs)) perturbation configs: $pert_configs")

    for case_name in case_names
        println("=== $case_name ===")
        file_pth = joinpath(NETWORK_INFO_DIR, case_name, "$case_name.m")
        local test_case
        try
            test_case = PowerModels.parse_file(file_pth)
            # solve_acpf = false: generate_data doesn't use the pv_pairs that
            # step computes, so skip the (potentially slow) nearest-gen search
            test_case = prepare_test_case(test_case, case_name, file_pth; solve_acpf = false)
        catch e
            println("  skipping $case_name: failed to load/prepare test case ($e)")
            continue
        end

        # baseline sanity check: the unperturbed case should itself converge
        # on AC-PF and have its nominal load within max_pd at every bus,
        # before spending time generating perturbed datapoints for a case
        # that's broken from the start -- e.g. case3022/dcfeas_pert: a bus
        # whose nominal (unperturbed) load already exceeds max_pd can never
        # produce a feasible perturbed sample either, no matter how small
        # the perturbation.
        #
        # AC-PF is tried with `grainger = true` first (the default everywhere
        # else in this pipeline); if that doesn't converge, it's retried with
        # `grainger = false` before giving up on the case entirely. Whichever
        # one converges is remembered as `use_grainger` and passed through to
        # every `generate_data` call below, so a case that only works without
        # the grainger technique gets its whole dataset generated that way
        # rather than silently skipped.
        baseline_ok = true
        use_grainger = true
        try
            baseline_case, max_pd = prepare_test_case_perturbations(deepcopy(test_case))
            max_gen = sum([gen["pmax"] for gen in values(baseline_case["gen"])])
            min_gen = sum([gen["pmin"] for gen in values(baseline_case["gen"])])
            loads = zeros(length(baseline_case["load"]))
            for load in values(baseline_case["load"])
                loads[load["index"]] = load["pd"]
            end
            if !_verify_loads_(baseline_case, loads, max_gen, min_gen, max_pd)
                println("  skipping $case_name: baseline (unperturbed) load exceeds max_pd at some bus")
                baseline_ok = false
            else
                baseline_res = PowerModels.compute_ac_pf(baseline_case, grainger = true, mapping = true)
                if !baseline_res["termination_status"]
                    println("  skipping $case_name: baseline (unperturbed) AC-PF did not converge ")
                    baseline_ok = false
                end
            end
        catch e
            println("  skipping $case_name: baseline AC-PF check errored ($e)")
            baseline_ok = false
        end
        baseline_ok || continue

        for pert_config_file in pert_configs
            pert_name = splitext(pert_config_file)[1]
            dataset_path = joinpath(TESTCASE_PATH, "data", case_name, pert_name)
            if skip_existing && isdir(dataset_path)
                println("  -- $pert_name -- skipping, $dataset_path already exists")
                continue
            end

            println("  -- $pert_name --")
            try
                generate_data(test_case, NUM_POINTS, case_name, pert_name;
                              pert_config_path = pert_config_file, log_level = log_level, grainger = use_grainger)
            catch e
                println("    generate_data failed for $case_name/$pert_name: $e")
                continue
            end

            try
                summary = analyze_dataset(dataset_path)
                _print_summary(summary)
                report_path = joinpath(dataset_path, "violation_summary.json")
                _write_report(summary, report_path)
                println("    saved violation summary to $report_path")
            catch e
                println("    solution_analysis failed for $case_name/$pert_name: $e")
            end
        end
    end
end

main()
