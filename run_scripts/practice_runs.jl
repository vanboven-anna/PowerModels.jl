include("../config.jl")
include("find_nearest_gens.jl")
include("generate_dataset.jl")
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Pkg.instantiate
using Revise
using PowerModels
push!(LOAD_PATH, DATA_PATH)
using Infiltrator
Infiltrator.toggle_async_check(false)
using DataFrames
using OrderedCollections 
using DataStructures
using JSON
using XLSX
using Ipopt
using Graphs 
using Infiltrator
using JuMP
using LinearAlgebra
using Random 
Random.seed!(1)


# pull in a test case 
case_name = "case14"
dataset_name = "dcfeas_pert"
file_pth = joinpath(DATA_PATH, "test_cases/data/$case_name/$dataset_name/0.json")
test_case = PowerModels.parse_file(file_pth)
test_case = prepare_test_case(test_case, case_name, file_pth)

#  compute ac opf setpoint dist 
ipopt = Ipopt.Optimizer
result = PowerModels.solve_dc_ac_pf(test_case, ipopt)


# # compute acpf 
# PowerModels.logger_config!("debug")
# nearest_gens = find_nearest_generators_khop(file_pth)
# test_case["pv_pairs"] = nearest_gens
# result = PowerModels.compute_ac_pf_mult_buses(test_case, grainger = true,  swap_technique = "nearest_gen", debug = true, obo = false)

