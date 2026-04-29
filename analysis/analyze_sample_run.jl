using Pkg
# Pkg.instantiate
Pkg.activate(joinpath(@__DIR__, ".."))
include("../config.jl")
include("../run_scripts/find_nearest_gens.jl")
include("../run_scripts/generate_dataset.jl")
include("../run_scripts/practice_runs.jl")
include("./plotting_functions.jl")

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


"Plot condition number and determinant of the jacobian over runs"
function plot_cond_det(results)
    jac_hist = parse_jacobian_history(results["jacobian_history"])
    cond_lst, det_lst = [], []
    for jac_iters in jac_hist
        iter_cond, iter_det = [], []
        for mat in jac_iters 
            push!(iter_cond, cond(mat))
            push!(iter_det, det(mat))
        end
        push!(cond_lst, iter_cond)
        push!(det_lst, iter_det)
    end
    # plot values 
    cond_plot = plot(title="condition number")
    cond_plot = plot_var_over_time(cond_lst, "condition number", cond_plot)
    det_plot = plot(title="determinant")
    det_plot = plot_var_over_time(det_lst, "determinant", det_plot)
    combined_figure = plot(cond_plot, det_plot, layout=(1, 2), size=(800, 400))
    display(combined_figure)
end

"Plot variable values over time"
function plot_var_vals(test_case, results)
    am = results["pf_data"].am
    vm_plot = plot(title="bus VMs", ylims = (0.9, 1.1))
    va_plot = plot(title="bus VAs", ylims = (-0.5, 0.1))
    qg_plot = plot(title="bus QGs", ylims = (-0.5, 0.5))
    x_hist = parse_jacobian_history(results["x_history"], is_mat = false)
    for bus_ind in keys(test_case["bus"])
        vm_plot = parse_var_vals(x_hist, results["solution_history"], results["mapping_dicts"], "vm", am.bus_to_idx[parse(Int64, bus_ind)], vm_plot)
        va_plot = parse_var_vals(x_hist, results["solution_history"], results["mapping_dicts"], "va", am.bus_to_idx[parse(Int64, bus_ind)], va_plot)
    end
    for gen_ind in keys(test_case["gen"])
        bus_ind = am.bus_to_idx[test_case["gen"][gen_ind]["gen_bus"]]
        gen_ind = parse(Int64, gen_ind)
        qg_plot = parse_var_vals(x_hist, results["solution_history"], results["mapping_dicts"], "q", bus_ind, qg_plot; 
                            label=test_case["gen"][string(gen_ind)]["gen_bus"], gen_ind = gen_ind)
    end
    combined_figure = plot(vm_plot, va_plot, qg_plot, layout = (2,2), size = (800, 800))
    display(combined_figure)
end

test_case, file_pth = create_test_case("case14", 1.53)
PowerModels.logger_config!("debug")
nearest_gens = find_nearest_generators_khop(file_pth)
test_case["pv_pairs"] = nearest_gens
result = PowerModels.compute_ac_pf_mult_buses(test_case, grainger = true,  swap_technique = "qv_inv", debug = true, obo = true)
# plot_cond_det(result);
plot_var_vals(test_case, result)

