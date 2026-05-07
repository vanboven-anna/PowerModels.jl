import LinearAlgebra 
using Infiltrator
Infiltrator.toggle_async_check(false)
using Test

function compute_jacobian!(J, x, pf_data, mapping_dict)
    # features from pf data 
    bus_type_idx, neighbors, am, vm_idx, va_idx = pf_data.bus_type_idx, pf_data.neighbors, pf_data.am, pf_data.vm_idx, pf_data.va_idx
    # functions for each type of derivative
    function dpdv(i)
        y_ii = am.matrix[i, i]
        return 2*real(y_ii)*vm_idx[i] + sum(  real(am.matrix[i,k]) * vm_idx[k] *  cos(va_idx[i] - va_idx[k]) + imag(am.matrix[i,k]) * vm_idx[k] * sin(va_idx[i] - va_idx[k]) for k in neighbors[i] if k != i)
    end
    function dqdv(i)
        y_ii = am.matrix[i, i]
        return -2*imag(y_ii)*vm_idx[i] + sum( -imag(am.matrix[i,k]) * vm_idx[k] *  cos(va_idx[i] - va_idx[k]) + real(am.matrix[i,k]) * vm_idx[k] * sin(va_idx[i] - va_idx[k]) for k in neighbors[i] if k != i)
    end
    function dpdtheta(i)
        return vm_idx[i] * sum(  real(am.matrix[i,k]) * vm_idx[k] * -sin(va_idx[i] - va_idx[k]) + imag(am.matrix[i,k]) * vm_idx[k] * cos(va_idx[i] - va_idx[k]) for k in neighbors[i] if k != i)
    end
    function dqdtheta(i)
        return vm_idx[i] * sum( -imag(am.matrix[i,k]) * vm_idx[k] * -sin(va_idx[i] - va_idx[k]) + real(am.matrix[i,k]) * vm_idx[k] * cos(va_idx[i] - va_idx[k]) for k in neighbors[i] if k != i)
    end
    function dpdp(i)
        return 1
    end
    function dqdp(i)
        return 0
    end
    function dpdq(i)
        return 0
    end
    function dqdq(i)
        return 1
    end
    function dpn_dv(i, j)
        y_ij = am.matrix[i,j]
        return vm_idx[i] * ( real(y_ij) * cos(va_idx[i] - va_idx[j]) + imag(y_ij) *  sin(va_idx[i] - va_idx[j]))
    end
    function dqn_dv(i, j)
        y_ij = am.matrix[i,j]
        return vm_idx[i] * (-imag(y_ij) * cos(va_idx[i] - va_idx[j]) + real(y_ij) *  sin(va_idx[i] - va_idx[j]))
    end
    function dpn_dtheta(i, j)
        y_ij = am.matrix[i,j]
        return vm_idx[i] * vm_idx[j] * ( real(y_ij) * sin(va_idx[i] - va_idx[j]) + imag(y_ij) * -cos(va_idx[i] - va_idx[j]))
    end
    function dqn_dtheta(i, j)
        y_ij = am.matrix[i,j]
        return vm_idx[i] * vm_idx[j] * (-imag(y_ij) * sin(va_idx[i] - va_idx[j]) + real(y_ij) * -cos(va_idx[i] - va_idx[j]))
    end

    # iterate through each power balance equation
    for i in eachindex(am.idx_to_bus)
        r_real = 2*i - 1
        r_imag = 2*i
        # iterate through each bus connected to get related variables 
        for j in neighbors[i]
            if bus_type_idx[j] == 1
                vm_col = mapping_dict[j]["vm"]
                va_col = mapping_dict[j]["va"]
                if i == j 
                    J[r_real, vm_col] = dpdv(j)
                    J[r_imag, vm_col] = dqdv(j)
                    J[r_real, va_col] = dpdtheta(j)
                    J[r_imag, va_col] = dqdtheta(j)
                else 
                    J[r_real, vm_col] = dpn_dv(i, j)
                    J[r_imag, vm_col] = dqn_dv(i, j)
                    J[r_real, va_col] = dpn_dtheta(i, j)
                    J[r_imag, va_col] = dqn_dtheta(i, j)
                end
            elseif bus_type_idx[j] == 2 
                q_col = mapping_dict[j]["q"]
                va_col = mapping_dict[j]["va"]
                if i == j 
                    J[r_real, q_col] = dpdq(j)
                    J[r_imag, q_col] = dqdq(j)
                    J[r_real, va_col] = dpdtheta(j)
                    J[r_imag, va_col] = dqdtheta(j)
                else 
                    J[r_real, q_col] = 0
                    J[r_imag, q_col] = 0
                    J[r_real, va_col] = dpn_dtheta(i, j)
                    J[r_imag, va_col] = dqn_dtheta(i, j)
                end
            elseif bus_type_idx[j] == 3
                q_col = mapping_dict[j]["q"]
                p_col = mapping_dict[j]["p"]
                if i == j 
                    J[r_real, q_col] = dpdq(j)
                    J[r_imag, q_col] = dqdq(j)
                    J[r_real, p_col] = dpdp(j)
                    J[r_imag, p_col] = dqdp(j)
                else 
                    J[r_real, q_col] = 0
                    J[r_imag, q_col] = 0
                    J[r_real, p_col] = 0
                    J[r_imag, p_col] = 0
                end   

            elseif bus_type_idx[j] == 5       
                va_col = mapping_dict[j]["va"]
                if i == j 
                    J[r_real, va_col] = dpdtheta(j)
                    J[r_imag, va_col] = dqdtheta(j)
                else 
                    J[r_real, va_col] = dpn_dtheta(i, j)
                    J[r_imag, va_col] = dqn_dtheta(i, j)
                end     
            elseif bus_type_idx[j] == 6
                q_col = mapping_dict[j]["q"]
                va_col = mapping_dict[j]["va"]
                vm_col = mapping_dict[j]["vm"]
                if i == j 
                    J[r_real, q_col] = dpdq(j)
                    J[r_imag, q_col] = dqdq(j)
                    J[r_real, va_col] = dpdtheta(j)
                    J[r_imag, va_col] = dqdtheta(j)
                    J[r_real, vm_col] = dpdv(j)
                    J[r_imag, vm_col] = dqdv(j)
                else 
                    J[r_real, q_col] = 0
                    J[r_imag, q_col] = 0
                    J[r_real, va_col] = dpn_dtheta(i, j)
                    J[r_imag, va_col] = dqn_dtheta(i, j)
                    J[r_real, vm_col] = dpn_dv(i, j)
                    J[r_imag, vm_col] = dqn_dv(i, j)
                end                                               
            end
        end
    end
end

@testset "testing the jacobian updates" begin 
    # pull in easy test case 
    test_case = PowerModels.parse_file(joinpath(@__DIR__, "data", "matpower","case5.m"))
    # re-set the vms to create violations 
    test_case["bus"]["2"]["vmax"] = 1.02
    test_case["bus"]["2"]["vmin"] = 1.0

    @testset "performing a single bus swap is a rank-one update to the jacobian" begin 
        init_results = PowerModels.compute_ac_pf(test_case, enforce_q_lims = false, mapping = true)
        J_init = init_results["jacobian_history"][end-1] # the final jacobian from the intial run 
        mapping_dict = init_results["mapping_dicts"][1] # mapping dict
        new_mapping = deepcopy(mapping_dict)
        pf_data = init_results["pf_data"] # data from initial run at convergence 
        #  perform a bus swap between PV bus 1 and PQ bus 2
        pf_data.bus_type_idx[1] = 6 
        pf_data.bus_type_idx[2] = 5 
        pf_data.vm_idx[2] = test_case["bus"]["2"]["vmin"] # set the vm to the lower limit (lower violation)
        new_mapping[1]["vm"] = mapping_dict[2]["vm"] # switch which vm is a variable
        new_mapping[2]["vm"] = 0
        # fill in the first iteration of the jacobian with this new variable and new setpoint 
        J_update = deepcopy(J_init)
        J_update.nzval .= 0.0
        compute_jacobian!(J_update, init_results["x_history"][end-1], pf_data, new_mapping)
        @infiltrate
        # compare columns between matrices to see how many experienced changes
        diff_mask = [sum(abs.(J_init[:, i] .- J_update[:, i])) > 1e-7 for i in 1:size(J_init, 2)]
        diff_cols = findall(diff_mask)
        @test length(diff_cols) == 1
    end
end