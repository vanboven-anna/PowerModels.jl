# generate + analyze every (test case, perturbation config) pair, skipping failures: julia --project=. run_scripts/generate_and_analyze_all.jl

include("./generate_dataset.jl")
include("../analysis_scripts/solution_analysis.jl")

const NETWORK_INFO_DIR = joinpath(DATA_PATH, "test_cases/network_info")
const NUM_POINTS = 10

"""
Bus count of a Matpower `.m` file, by regex rather than a full parse.
`typemax(Int)` when the `mpc.bus` block isn't found, so it sorts last.
"""
function _bus_count(m_path::AbstractString)
    txt = read(m_path, String)
    m = match(r"mpc\.bus\s*=\s*\[(.*?)\n[ \t]*\]"s, txt)
    m === nothing && return typemax(Int)
    return count(split(m.captures[1], '\n')) do line
        s = strip(line)
        !isempty(s) && !startswith(s, "%")
    end
end

"""
Case names in `network_info_dir` holding a `<case_name>/<case_name>.m`,
sorted by bus count, smallest first.
"""
function discover_case_names(network_info_dir = NETWORK_INFO_DIR)
    case_names = String[]
    for entry in sort(readdir(network_info_dir))
        case_dir = joinpath(network_info_dir, entry)
        if isdir(case_dir) && isfile(joinpath(case_dir, "$entry.m"))
            push!(case_names, entry)
        end
    end
    return sort(case_names; by = c -> (_bus_count(joinpath(network_info_dir, c, "$c.m")), c))
end

"List the perturbation config filenames (e.g. `default_pert.json`) in `PERTURBATION_CONFIGS_DIR`."
function discover_pert_configs()
    return sort(filter(f -> endswith(f, ".json"), readdir(PERTURBATION_CONFIGS_DIR)))
end

"""
Full case x config sweep, smallest network first.
`start_case`: bus-count threshold, not an alphabetical one.
`quiet`: silence PowerModels Info/Warn.
`skip_existing`: skip a pair whose output folder exists (existence, not completeness).
"""
function main(; start_case::Union{Nothing,AbstractString} = nothing,
                quiet::Bool = true,
                config_lst = nothing,
                skip_existing::Bool = true)
    log_level = quiet ? "error" : "warn"
    PowerModels.logger_config!(log_level)

    # optimizer for the DC-OPF warm-start fallback in the baseline check
    ipopt_print_level = log_level == "error" ? 0 : 1
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => ipopt_print_level)

    # sorted smallest bus count first
    case_names = discover_case_names()
    if start_case !== nothing
        start_case in case_names ||
            error("start_case = \"$start_case\" not found among discovered case names: $case_names")
        min_buses = _bus_count(joinpath(NETWORK_INFO_DIR, start_case, "$start_case.m"))
        case_names = [c for c in case_names
                      if _bus_count(joinpath(NETWORK_INFO_DIR, c, "$c.m")) >= min_buses]
    end
    pert_configs = isnothing(config_lst) ? discover_pert_configs() : config_lst
    println("Found $(length(case_names)) test cases and $(length(pert_configs)) perturbation configs: $pert_configs")
    println("Case order (by bus count): $case_names")

    for case_name in case_names
        println("=== $case_name ===")
        file_pth = joinpath(NETWORK_INFO_DIR, case_name, "$case_name.m")
        local test_case
        try
            test_case = PowerModels.parse_file(file_pth)
            # solve_acpf = false: generate_data ignores pv_pairs, skip the nearest-gen search
            test_case = prepare_test_case(test_case, case_name, file_pth; solve_acpf = false)
        catch e
            println("  skipping $case_name: failed to load/prepare test case ($e)")
            continue
        end

        # baseline check: a case broken unperturbed can't produce feasible samples
        baseline_ok = true
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
                # enforce_q_lims = false: PV->PQ switching often fails to converge here
                baseline_res = PowerModels.compute_ac_pf(baseline_case, grainger = true,
                                                         mapping = true, enforce_q_lims = false)
                if !baseline_res["termination_status"]
                    # a flat vm/va start Newton-Raphson can't solve from; retry off DC-OPF
                    dc = try
                        PowerModels.solve_dc_opf(baseline_case, ipopt)
                    catch e
                        println("  $case_name: baseline DC-OPF (for warm start) errored ($e)")
                        nothing
                    end
                    if dc !== nothing && dc["termination_status"] == LOCALLY_SOLVED
                        warmstart_pf_from_dc!(baseline_case, dc["solution"])
                        baseline_res = PowerModels.compute_ac_pf(baseline_case, grainger = true,
                                                                 mapping = true, enforce_q_lims = false)
                        if baseline_res["termination_status"]
                            # carry the warm start into generate_data's attempts too
                            warmstart_pf_from_dc!(test_case, dc["solution"])
                            println("  $case_name: baseline AC-PF converged after DC-OPF warm start")
                        end
                    end
                    if !baseline_res["termination_status"]
                        println("  skipping $case_name: baseline (unperturbed) AC-PF did not converge (even with a DC-OPF warm start)")
                        baseline_ok = false
                    end
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
                              pert_config_path = pert_config_file, log_level = log_level, grainger = true)
            catch e
                println("    generate_data failed for $case_name/$pert_name: $e")
                continue
            end

            try
                summary = analyze_dataset(dataset_path)
                _print_summary(summary)
                # put the report next to the datapoints analyze_dataset found
                report_path = joinpath(summary["dataset_path"], "violation_summary.json")
                _write_report(summary, report_path)
                println("    saved violation summary to $report_path")
            catch e
                println("    solution_analysis failed for $case_name/$pert_name: $e")
            end
        end
    end
end

main(;start_case="case2869_pegase")
