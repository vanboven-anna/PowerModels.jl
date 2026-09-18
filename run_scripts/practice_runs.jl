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
dataset_name = "extreme_pert"
# one datapoint out of dataset.h5; the rest are never read
dset_dir = joinpath(DATA_PATH, "test_cases/data/$case_name/$dataset_name/baseline_acpf")
test_case = load_datapoint(dset_dir, first(dataset_datapoints(dset_dir)))
test_case = prepare_test_case(test_case, case_name, dset_dir)

#  compute ac opf setpoint dist (the device=true pipeline lives in generate_dataset.jl)
ipopt = Ipopt.Optimizer
solve_dc_ac_pf!(test_case, ipopt; device = true)


#  compute acpf 
# PowerModels.logger_config!("debug")
# nearest_gens = find_nearest_generators_khop(file_pth)
# test_case["pv_pairs"] = nearest_gens
# result = PowerModels.compute_ac_pf_mult_buses(test_case, grainger = true,  swap_technique = "nearest_gen", debug = true, obo = false)

