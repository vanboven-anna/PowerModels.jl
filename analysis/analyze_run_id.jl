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

function count_successful_swaps(test_case, bus_df)
    num_dps = unique(bus_df["datapoint"])
    bus_swap_rates = Dict{String, Float32{}}(bus_ind => [0 for _ in 1:num_dps] for 
                                                bus_ind in keys(test_case["bus"]))
    for bus_ind in keys(test_case["bus"])
        for dp in num_dps
            # get datapoint rows 
            dp_df = filter(row -> row["datapoint"] == dp, bus_df)
            # see how many times a bus became a P or PQV
            bus_types = dp_df["bt_$bus_ind"]
            t6_switched = sum(((bus_types .== 6) .& (circshift(bus_types, -1) .!= 6))[1:end-1])
            t5_switched =sum(((bus_types .== 5) .& (circshift(bus_types, -1) .!= 5))[1:end-1])
            bus_swap_rates[bus_ind][dp] = t6_switched + t5_switched
        end
    end
    swap_dict = DataFrame(bus_swap_rates)
end



# pull in test case data
case_name = "case14"
file_pth = joinpath(DATA_PATH, "test_cases/network_info/$case_name/$(case_name).m")
test_case = PowerModels.parse_file(file_pth)
test_case = prepare_test_case(test_case, case_name, file_pth)
delta = 1.53
filename = joinpath(RESULTS_PATH, "$(case_name)/$delta.xlsx")
run_info = DataFrame(XLSX.readtable(filename, "run_info"))
solns = DataFrame(XLSX.readtable(filename, "solns"))
bus_types = DataFrame(XLSX.readtable(filename, "bus_types"))
violations = DataFrame(XLSX.readtable(filename, "violations"))
