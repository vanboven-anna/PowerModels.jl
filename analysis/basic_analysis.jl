using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include("../run_scripts/generate_dataset.jl")
include("../config.jl")
# Pkg.instantiate
using Revise
push!(LOAD_PATH, DATA_PATH)
using Infiltrator
Infiltrator.toggle_async_check(false)
using DataFrames
using DataStructures
using JSON
using XLSX
using Distributions
using Random 
Random.seed!(2)

function analyze_dataset(combined_df)
    analysis_dict = Dict{String, Float64}("time" => 0, "swap_iters" => 0, "jac_iters" => 0, "total_violations" => 0, "total_vm_violations" => 0, 
                "total_q_violations" => 0, "total_branch_violations" => 0, "total_violation_mag" => 0, "total_vm_violation_mag" => 0, 
                "total_q_violation_mag" => 0, "total_branch_violation_mag" => 0)
    for col in keys(analysis_dict)
        try
            analysis_dict[col] = sum(combined_df[!, col])/nrow(combined_df)
        catch e 
            @infiltrate
            rethrow(e)
        end
    end
    return analysis_dict
end

function combine_datasets(run_info, violations, run_id)
    parse_run = filter("run_id" => val -> val == run_id, run_info)[:, ["datapoint", "time", "swap_iters", "jac_iters"]]
    # grab the last run from the violations 
    parse_viol = combine(groupby(violations, ["datapoint", "run_id"])) do sdf
        sdf[argmax(sdf[!, "iter"]), :]
    end
    parse_viol = filter("run_id" => val -> val == run_id, parse_viol)[:, ["datapoint", "total_violations", "total_vm_violations", 
                "total_q_violations", "total_branch_violations", "total_violation_mag", "total_vm_violation_mag", 
                "total_q_violation_mag", "total_branch_violation_mag"]]
    combined_df = innerjoin(parse_run, parse_viol, on = "datapoint")
    return combined_df
end

function parse_run_ids(run_info, run_id)
    run_row = filter(row -> row["run_id"] == run_id, run_info)[1,:]
    run_dict = Dict(i => run_row[i] for i in ["pf_type", "obo", "grainger", "swap_technique"])
    return run_dict
end 

function export_analysis(analyses, write_out, filename, print_out)
    if write_out 
        dir_path = dirname(filename)
        mkpath(dir_path)
        if isfile(filename)
            rm(filename)
        end
        XLSX.openxlsx(filename, mode="w") do xf
            sheet1 = XLSX.addsheet!(xf, "basic_analysis")
            XLSX.writetable!(sheet1, Tables.columntable(analyses))
        end
    end
    if print_out 
        col_width = 22
        
        baseline_cols = ["grainger", "time", "jac_iters", "total_violation_mag", "total_violations"]
        println("-"^40 * " run results baseline " * "-"^40)
        curr_df = filter(row -> row["pf_type"] == "baseline", analyses)
        println(join([rpad(col, col_width) for col in baseline_cols]))
        grainger = first(filter(row -> row["grainger"] == 1, curr_df))
        println(join([rpad(string(grainger[col]), col_width) for col in baseline_cols]))
        no_grainger = first(filter(row -> row["grainger"] == 0, curr_df))
        println(join([rpad(string(no_grainger[col]), col_width) for col in baseline_cols]))
        println("-"^100)

        qlim_cols = ["grainger", "obo", "time","jac_iters", "swap_iters","total_violation_mag", "total_violations"]
        println("-"^40 * " run results qlim " * "-"^40)
        curr_df = filter(row -> row["pf_type"] == "qlim", analyses)
        println(join([rpad(col, col_width) for col in qlim_cols]))
        grainger = filter(row -> row["grainger"] == 1, curr_df)
        no_grainger = filter(row -> row["grainger"] == 0, curr_df)
        for test in [grainger, no_grainger]
            obo = first(filter(r -> r["obo"] == 1, test))
            obo_off = first(filter(r -> r["obo"] == 0, test))
            println(join([rpad(string(obo[col]), col_width) for col in qlim_cols]))
            println(join([rpad(string(obo_off[col]), col_width) for col in qlim_cols]))
        end
        println("-"^100)

        mbuses_cols = ["grainger", "obo", "swap_technique", "time","jac_iters", "swap_iters","total_violation_mag", "total_violations"]
        println("-"^40 * " run results mbuses " * "-"^40)
        curr_df = filter(row -> row["pf_type"] == "mbuses", analyses)
        println(join([rpad(col, col_width) for col in mbuses_cols]))
        grainger = filter(row -> row["grainger"] == 1, curr_df)
        no_grainger = filter(row -> row["grainger"] == 0, curr_df)
        for test in [grainger, no_grainger]
            obo = filter(r -> r["obo"] == 1, test)
            obo_off = filter(r -> r["obo"] == 0, test)
            for obo_row in [obo, obo_off]
                st1 = first(filter(r -> r["swap_technique"] == "qv_inv", obo_row))
                st2 = first(filter(r -> r["swap_technique"] == "nearest_gen", obo_row))
                for st_row in [st1, st2]
                    println(join([rpad(string(st_row[col]), col_width) for col in mbuses_cols]))
                end
            end
        end
        println("-"^100)

    end
end


function perform_analysis(run_info, violations; write_out = false, filename = nothing, print_out = true)
    analysis_df = DataFrame()
    for run_id in 1:14
        combined_df = combine_datasets(run_info, violations, run_id)
        run_analyses = analyze_dataset(combined_df)
        run_dict = parse_run_ids(run_info, run_id)
        curr_df = hcat(DataFrame(run_dict), DataFrame(run_analyses))
        append!(analysis_df, curr_df)
    end
    return analysis_df
end


# pull in test case data
case_name = "case57"
file_pth = joinpath(DATA_PATH, "test_cases/network_info/$case_name/$(case_name).m")
test_case = PowerModels.parse_file(file_pth)
test_case = prepare_test_case(test_case, case_name, file_pth)
delta = 0.34
filename = joinpath(RESULTS_PATH, "$(case_name)/$delta.xlsx")
run_info = DataFrame(XLSX.readtable(filename, "run_info"))
solns = DataFrame(XLSX.readtable(filename, "solns"))
bus_types = DataFrame(XLSX.readtable(filename, "bus_types"))
violations = DataFrame(XLSX.readtable(filename, "violations"))
analysis_df = perform_analysis(run_info, violations; write_out = true, filename = joinpath(RESULTS_PATH, "$(case_name)/$(delta)_basic_analysis.xlsx")); 




