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

"number of times that each bus switched bus type for each datapoint"
function count_successful_swaps(test_case, bus_df)
    num_dps = unique(bus_df[!, "datapoint"])
    bus_swap_rates = Dict{String, Vector{Float64}}(bus_ind => [0 for _ in num_dps] for 
                                                bus_ind in keys(test_case["bus"]))
    for bus_ind in keys(test_case["bus"])
        for dp in num_dps
            # get datapoint rows 
            dp_df = filter(row -> row["datapoint"] == dp, bus_df)
            # see how many times a bus switches its bus type
            bus_types = dp_df[!, "bt_$bus_ind"]
            bus_swap_rates[bus_ind][Int(dp + 1)] = sum(bus_types[1:end-1] .!= bus_types[2:end], init = 0)
        end
    end
    swap_dict = DataFrame(bus_swap_rates)
    return swap_dict
end

"average number of jacobian iterations and P-PQV pairs"
function count_iter_bs(test_case, bus_df, soln_df)
    num_dps = unique(bus_df[!, "datapoint"])
    dp_jac, dp_swap = [], []
    for dp in num_dps 
        dp_df = filter(row -> row["datapoint"] == dp, bus_df)
        iters = unique(dp_df[!, "iter"])
        iter_jac, iter_swap = [], []
        # iterate through each swap iteration
        for iter in iters 
            iter_df = filter(row -> row["iter"] == iter, dp_df)
            bus_types = iter_df[!, ["bt_$bus_ind" for bus_ind in keys(test_case["bus"])]]
            num_swaps = sum([1 for bt in first(bus_types) if bt == 6], init = 0)
            num_jacs = filter(row -> (row["datapoint"] == dp) && (row["iter"] == iter), soln_df)[!, "jac_iter"][1]
            push!(iter_jac, num_jacs)
            push!(iter_swap, num_swaps)
        end
        push!(dp_jac, mean(iter_jac))
        push!(dp_swap, mean(iter_swap))
    end
    return dp_jac, dp_swap
end

"average summed change in solution setpoint between iteration, considering all variables"
function compare_soln_iters(soln_df)
    num_dps = unique(soln_df[!, "datapoint"])
    dp_soln = []
    for dp in num_dps
        dp_df = filter(row -> row["datapoint"] == dp, soln_df)
        soln_diff = []
        for row in eachrow(dp_df)
            next_iter = row["iter"] + 1
            try
                # get current and next solution
                next_soln = first(filter(row -> row["iter"] == next_iter, dp_df))
                curr_soln = values(row[setdiff(names(dp_df), ["datapoint", "run_id", "iter", "jac_iter", "final_iter"])])
                next_soln =  values(next_soln[setdiff(names(dp_df), ["datapoint", "run_id", "iter", "jac_iter", "final_iter"])])
                # compare solution values 
                diff = sum([abs(i - j) for (i,j) in zip(curr_soln, next_soln)], init = 0)
                push!(soln_diff, diff)
            catch e 
                if e isa BoundsError
                    break 
                else 
                    rethrow(e)
                end
            end
        end
        push!(dp_soln, length(soln_diff) > 0 ? mean(soln_diff) : 0)
    end
    return dp_soln
end

"analysis"
function compute_analysis(case_name, delta)
    file_pth = joinpath(DATA_PATH, "test_cases/network_info/$case_name/$(case_name).m")
    test_case = PowerModels.parse_file(file_pth)
    test_case = prepare_test_case(test_case, case_name, file_pth)
    filename = joinpath(RESULTS_PATH, "$(case_name)/$delta.xlsx")
    solns = DataFrame(XLSX.readtable(filename, "solns"))
    bus_types = DataFrame(XLSX.readtable(filename, "bus_types"))
    run_ids = unique(solns[!, "run_id"])
    # prepare for writing out 
    filename = joinpath(RESULTS_PATH, "$case_name/$(delta)_run_ids.xlsx")
    dir_path = dirname(filename)
    mkpath(dir_path)
    if isfile(filename)
        rm(filename)
    end
    XLSX.openxlsx(filename, mode="w") do xf
        for id in run_ids
            println("analyzing Run $id...")
            bt_runid = filter(row -> row["run_id"] == id, bus_types)
            soln_runid = filter(row -> row["run_id"] == id, solns)
            swap_df = count_successful_swaps(test_case, bt_runid)
            avg_jac, avg_pairs = count_iter_bs(test_case, bt_runid, soln_runid)
            swap_df[!, "avg_jac"] = avg_jac
            swap_df[!, "avg_pairs"] = avg_pairs
            swap_df[!, "soln_diff"] = compare_soln_iters(soln_runid) 
            swap_df[!, "datapoint"] = unique(soln_runid[!, "datapoint"])
            # write out run id 
            sheet1 = XLSX.addsheet!(xf, string(id))
            XLSX.writetable!(sheet1, Tables.columntable(swap_df))
        end
    end
end


compute_analysis("case57", 0.34)




# case_name = "case14"
# delta = 1.53
# file_pth = joinpath(DATA_PATH, "test_cases/network_info/$case_name/$(case_name).m")
# test_case = PowerModels.parse_file(file_pth)
# test_case = prepare_test_case(test_case, case_name, file_pth)
# filename = joinpath(RESULTS_PATH, "$(case_name)/$delta.xlsx")
# solns = DataFrame(XLSX.readtable(filename, "solns"))
# bus_types = DataFrame(XLSX.readtable(filename, "bus_types"))
# run_ids = unique(solns[!, "run_id"])
# id = 1
# bt_runid = filter(row -> row["run_id"] == id, bus_types)
# soln_runid = filter(row -> row["run_id"] == id, solns)
# avg_jac, avg_pairs = count_iter_bs(test_case, bt_runid, soln_runid)



