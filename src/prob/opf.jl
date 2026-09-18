debug = false
""
function solve_ac_opf(file, optimizer; kwargs...)
    return solve_opf(file, ACPPowerModel, optimizer; kwargs...)
end

function solve_dc_opf(file, optimizer; kwargs...)
    return solve_opf(file, DCPPowerModel, optimizer; kwargs...)
end

function solve_opf(file, model_type::Type, optimizer; kwargs...)
    return solve_model(file, model_type, optimizer, build_opf; kwargs...)
end

function solve_relaxed_opf(file, optimizer; kwargs...)
    return solve_model(file, ACPPowerModel, optimizer, build_relaxed_opf; kwargs...)
end

function solve_dc_ac_pf(file, optimizer; kwargs...)
    return solve_model(file, ACPPowerModel, optimizer, build_dc_ac_pf; kwargs...)
end

function solve_dc_ac_device_pf(file, optimizer; kwargs...)
    return solve_model(file, ACPPowerModel, optimizer, build_dc_ac_device_pf; kwargs...)
end

"""
    build_opf(pm::AbstractPowerModel)
"""
function build_opf(pm::AbstractPowerModel)
    variable_bus_voltage(pm)
    variable_gen_power(pm)
    variable_branch_power(pm)
    variable_dcline_power(pm)

    objective_min_fuel_and_flow_cost(pm)

    constraint_model_voltage(pm)

    for i in ids(pm, :ref_buses)
        constraint_theta_ref(pm, i)
    end

    for i in ids(pm, :bus)
        constraint_power_balance(pm, i)
    end

    for i in ids(pm, :branch)
        constraint_ohms_yt_from(pm, i)
        constraint_ohms_yt_to(pm, i)

        constraint_voltage_angle_difference(pm, i)

        constraint_thermal_limit_from(pm, i)
        constraint_thermal_limit_to(pm, i)
    end

    for i in ids(pm, :dcline)
        constraint_dcline_power_losses(pm, i)
    end
end


const DC_AC_PF_SOFT_BOUND_PENALTY = 1e3

"""
    build_dc_ac_pf for voltage setpoint minimization

Given a fixed active-power dispatch (`pg` pinned exactly to `pg_start` for
every non-slack generator) and initial generator voltage setpoints (`vg`),
finds the AC-feasible operating point that minimizes the total squared change
to those voltage setpoints. Generator reactive-power limits (qmin/qmax) and
bus voltage-magnitude limits (vmin/vmax) are kept soft: violable, but at a
very high penalty (`DC_AC_PF_SOFT_BOUND_PENALTY`), so the model stays
solvable (rather than becoming infeasible outright) if no fully-in-bounds
AC-feasible point exists near the given setpoints.
"""
function build_dc_ac_pf(pm::AbstractPowerModel)
    vm, va = variable_bus_voltage(pm; bounded = false)
    pg, pg_sps = variable_gen_power_real(pm)
    qg = variable_gen_power_imaginary(pm; bounded = false)
    variable_branch_power(pm)
    variable_dcline_power(pm)

    vm_sps = []
    vm_gens = []

    for i in ids(pm, :gen)
        push!(vm_sps, pm.data["gen"][string(i)]["vg"])
        push!(vm_gens, var(pm, nw_id_default, :vm, pm.data["gen"][string(i)]["gen_bus"]))
    end

    constraint_model_voltage(pm)

    # hard generator setpoint: pg is fixed exactly at its target value
    for (pg_i, sp) in zip(pg, pg_sps)
        constraint_pbal_sp(pm, pg_i, sp)
    end

    for i in ids(pm, :ref_buses)
        constraint_theta_ref(pm, i)
    end

    for i in ids(pm, :bus)
        constraint_power_balance(pm, i)
    end

    for i in ids(pm, :branch)
        constraint_ohms_yt_from(pm, i)
        constraint_ohms_yt_to(pm, i)

        constraint_voltage_angle_difference(pm, i)

        constraint_thermal_limit_from(pm, i)
        constraint_thermal_limit_to(pm, i)
    end

    for i in ids(pm, :dcline)
        constraint_dcline_power_losses(pm, i)
    end

    # objective: minimize the change to the generator voltage setpoints
    objective_min_vm_dist(pm, vm_gens, vm_sps)

    # soft qg/vm bounds: build them as ordinary hard constraints, then use
    # JuMP's PenaltyRelaxation (as in build_relaxed_opf) to turn them into
    # soft constraints, adding penalty*slack terms on top of the objective
    # set above rather than replacing it.

    soft_bound_penalties = Dict{JuMP.MOI.ConstraintIndex, Float64}()
    slack_bus = [bus["bus_i"] for (i, bus) in ref(pm, :bus) if bus["bus_type"] == 3][1]
    for (i, gen) in ref(pm, :gen)
        qg_ub = JuMP.@constraint(pm.model, 1.0 * qg[i] <= gen["qmax"])
        qg_lb = JuMP.@constraint(pm.model, 1.0 * qg[i] >= gen["qmin"])
        soft_bound_penalties[JuMP.index(qg_ub)] = DC_AC_PF_SOFT_BOUND_PENALTY
        soft_bound_penalties[JuMP.index(qg_lb)] = DC_AC_PF_SOFT_BOUND_PENALTY
    end
    for (i, bus) in ref(pm, :bus)
        vm_ub = JuMP.@constraint(pm.model, 1.0 * vm[i] <= bus["vmax"])
        vm_lb = JuMP.@constraint(pm.model, 1.0 * vm[i] >= bus["vmin"])
        soft_bound_penalties[JuMP.index(vm_ub)] = DC_AC_PF_SOFT_BOUND_PENALTY
        soft_bound_penalties[JuMP.index(vm_lb)] = DC_AC_PF_SOFT_BOUND_PENALTY
    end
    relaxation_penalty = JuMP.MOI.Utilities.PenaltyRelaxation(soft_bound_penalties; default = nothing)
    pm.data["soft_bound_penalty_dict"] = JuMP.MOI.modify(JuMP.backend(pm.model), relaxation_penalty)
end

"""
    build_dc_ac_device_pf for voltage AND device-setpoint minimization

Same skeleton as `build_dc_ac_pf` (fixed `pg`, soft `qg`/`vm` bounds, minimize
generator voltage-setpoint change) but also treats a subset of the network's
tunable devices as decision variables, so their setpoints can move too:
  - transformer tap ratio, for every branch with `branch["is_tap"] == true`
  - transformer phase shift, for every branch with `branch["is_shift"] == true`
  - shunt susceptance (`bs`), for every shunt

PowerModels' stock `constraint_ohms_yt_from`/`constraint_ohms_yt_to`/
`constraint_power_balance` treat tap/shift/shunt-bs as fixed data baked
directly into the nonlinear flow expressions, so they can't be reused as-is
here -- `_device_pf_ohms_yt_from!`/`_device_pf_ohms_yt_to!`/
`_device_pf_power_balance!` above are this function's own versions of those
three constraints, matching `form/acp.jl`'s ACP implementations term-for-term
except that a flagged branch/shunt's `tr`/`ti`/`tm`/`bs` are the JuMP
variable created for it rather than its fixed data value. A branch that
isn't `is_tap`/`is_shift` keeps its fixed `tap`/`shift`, same as always.

The objective adds a squared-distance-from-setpoint term for each of
tap/shift/bs on top of the generator voltage-setpoint term
(`objective_min_vm_dist`). `tap`/`shift` are bounded by
`branch["tmin"]/["tmax"]` and `branch["smin"]/["smax"]` when present, and
left unbounded otherwise; `bs` is bounded by `shunt["bmin"]/["bmax"]` when
present, unbounded otherwise. Unlike the `qg`/`vm` bounds inherited from
`build_dc_ac_pf`, these are hard bounds, not soft/penalized ones.

The solved `tap`/`shift`/`bs` values are written into the result's
`"solution"` dict (`solution["branch"][i]["tap"/"shift"]`,
`solution["shunt"][i]["bs"]`) alongside the usual `pf`/`qf`/`vm`/etc., so
callers can read the continuous device setpoints straight off a successful
solve.
"""
function build_dc_ac_device_pf(pm::AbstractPowerModel)
    vm, va = variable_bus_voltage(pm; bounded = false)
    pg, pg_sps = variable_gen_power_real(pm)
    qg = variable_gen_power_imaginary(pm; bounded = false)
    variable_branch_power(pm)
    variable_dcline_power(pm)

    vm_sps = []
    vm_gens = []
    for i in ids(pm, :gen)
        push!(vm_sps, pm.data["gen"][string(i)]["vg"])
        push!(vm_gens, var(pm, nw_id_default, :vm, pm.data["gen"][string(i)]["gen_bus"]))
    end

    # device variables: tap/shift only for flagged branches, bs for every shunt
    tap_var = Dict{Int, Any}()
    tap_sps = Dict{Int, Float64}()
    shift_var = Dict{Int, Any}()
    shift_sps = Dict{Int, Float64}()
    for (i, branch) in ref(pm, :branch)
        if get(branch, "is_tap", false)
            v = JuMP.@variable(pm.model, base_name = "tap_$i", start = branch["tap"])
            if haskey(branch, "tmin") && haskey(branch, "tmax")
                JuMP.set_lower_bound(v, branch["tmin"])
                JuMP.set_upper_bound(v, branch["tmax"])
            end
            tap_var[i] = v
            tap_sps[i] = branch["tap"]
        end
        if get(branch, "is_shift", false)
            v = JuMP.@variable(pm.model, base_name = "shift_$i", start = branch["shift"])
            if haskey(branch, "smin") && haskey(branch, "smax")
                JuMP.set_lower_bound(v, branch["smin"])
                JuMP.set_upper_bound(v, branch["smax"])
            end
            shift_var[i] = v
            shift_sps[i] = branch["shift"]
        end
    end

    bs_var = Dict{Int, Any}()
    bs_sps = Dict{Int, Float64}()
    for (i, shunt) in ref(pm, :shunt)
        v = JuMP.@variable(pm.model, base_name = "bs_$i", start = shunt["bs"])
        if haskey(shunt, "bmin") && haskey(shunt, "bmax")
            JuMP.set_lower_bound(v, shunt["bmin"])
            JuMP.set_upper_bound(v, shunt["bmax"])
        end
        bs_var[i] = v
        bs_sps[i] = shunt["bs"]
    end

    # expose the solved device setpoints in the result, same convention the
    # stock `variable_*` functions use (see `sol_component_value` above)
    for (i, v) in tap_var
        sol(pm, nw_id_default, :branch, i)["tap"] = v
    end
    for (i, v) in shift_var
        sol(pm, nw_id_default, :branch, i)["shift"] = v
    end
    for (i, v) in bs_var
        sol(pm, nw_id_default, :shunt, i)["bs"] = v
    end

    constraint_model_voltage(pm)

    # hard generator setpoint: pg is fixed exactly at its target value
    for (pg_i, sp) in zip(pg, pg_sps)
        constraint_pbal_sp(pm, pg_i, sp)
    end

    for i in ids(pm, :ref_buses)
        constraint_theta_ref(pm, i)
    end

    for i in ids(pm, :bus)
        _device_pf_power_balance!(pm, i, bs_var)
    end

    for (i, branch) in ref(pm, :branch)
        tm = get(branch, "is_tap", false) ? tap_var[i] : branch["tap"]
        shift = get(branch, "is_shift", false) ? shift_var[i] : branch["shift"]
        tr = tm * cos(shift)
        ti = tm * sin(shift)

        _device_pf_ohms_yt_from!(pm, i, tr, ti, tm)
        _device_pf_ohms_yt_to!(pm, i, tr, ti, tm)

        constraint_voltage_angle_difference(pm, i)

        constraint_thermal_limit_from(pm, i)
        constraint_thermal_limit_to(pm, i)
    end

    for i in ids(pm, :dcline)
        constraint_dcline_power_losses(pm, i)
    end

    # objective: minimize the change to the generator voltage setpoints,
    # plus the change to every tap/shift/bs setpoint
    objective_min_vm_dist(pm, vm_gens, vm_sps)
    device_dist = sum((tap_sps[i] - tap_var[i])^2 for i in keys(tap_var); init = 0.0) +
                  sum((shift_sps[i] - shift_var[i])^2 for i in keys(shift_var); init = 0.0) +
                  sum((bs_sps[i] - bs_var[i])^2 for i in keys(bs_var); init = 0.0)
    JuMP.set_objective_function(pm.model, JuMP.objective_function(pm.model) + device_dist)

    # soft qg/vm bounds -- see build_dc_ac_pf for the full explanation of
    # both gotchas (forcing affine form, and default = nothing)
    soft_bound_penalties = Dict{JuMP.MOI.ConstraintIndex, Float64}()
    slack_bus = [bus["bus_i"] for (i, bus) in ref(pm, :bus) if bus["bus_type"] == 3][1]
    for (i, gen) in ref(pm, :gen)
        gen["gen_bus"] == slack_bus && continue
        qg_ub = JuMP.@constraint(pm.model, 1.0 * qg[i] <= gen["qmax"])
        qg_lb = JuMP.@constraint(pm.model, 1.0 * qg[i] >= gen["qmin"])
        soft_bound_penalties[JuMP.index(qg_ub)] = DC_AC_PF_SOFT_BOUND_PENALTY
        soft_bound_penalties[JuMP.index(qg_lb)] = DC_AC_PF_SOFT_BOUND_PENALTY
    end
    for (i, bus) in ref(pm, :bus)
        vm_ub = JuMP.@constraint(pm.model, 1.0 * vm[i] <= bus["vmax"])
        vm_lb = JuMP.@constraint(pm.model, 1.0 * vm[i] >= bus["vmin"])
        soft_bound_penalties[JuMP.index(vm_ub)] = DC_AC_PF_SOFT_BOUND_PENALTY
        soft_bound_penalties[JuMP.index(vm_lb)] = DC_AC_PF_SOFT_BOUND_PENALTY
    end
    relaxation_penalty = JuMP.MOI.Utilities.PenaltyRelaxation(soft_bound_penalties; default = nothing)
    pm.data["soft_bound_penalty_dict"] = JuMP.MOI.modify(JuMP.backend(pm.model), relaxation_penalty)
end

"""
    build_ac_opf with relaxed constraints
"""
function build_relaxed_opf(pm::AbstractPowerModel)
    variable_bus_voltage(pm)
    pgs, pg_sps, qgs = variable_gen_power(pm)
    variable_branch_power(pm)
    variable_dcline_power(pm)

    constraint_dict = Dict()
    vm_dict = constraint_model_voltage(pm)
    constraint_dict =  isnothing(vm_dict) ? constraint_dict : merge(constraint_dict, vm_dict)
    for i in ids(pm, :ref_buses)
        tref = constraint_theta_ref(pm, i)
        constraint_dict[JuMP.index(tref)] = "tref_$i"
    end

    for i in ids(pm, :bus)
        constr_p, constr_q = constraint_power_balance(pm, i)
        constraint_dict[JuMP.index(constr_p)] = "pbal_$i"
        constraint_dict[JuMP.index(constr_q)] = "qbal_$i"
    end

    for i in ids(pm, :branch)
        ytfp, ytfq = constraint_ohms_yt_from(pm, i)
        constraint_dict[JuMP.index(ytfp)] = "ytfp_$i"
        constraint_dict[JuMP.index(ytfq)] = "ytfq_$i"
        yttp, yttq = constraint_ohms_yt_to(pm, i)
        constraint_dict[JuMP.index(yttp)] = "yttp_$i"
        constraint_dict[JuMP.index(yttq)] = "yttq_$i"

        vadiff = constraint_voltage_angle_difference(pm, i)
        constraint_dict[JuMP.index(vadiff)] = "vadiff_$i"

        tlf = constraint_thermal_limit_from(pm, i)
        tlt = constraint_thermal_limit_to(pm, i)
        constraint_dict[JuMP.index(tlf)] = "tlf_$i"
        constraint_dict[JuMP.index(tlt)] = "tlt_$i"
    end

    for i in ids(pm, :dcline)
        dcline = constraint_dcline_power_losses(pm, i)
        constraint_dict[JuMP.index(dcline)] = "dcline_$i"
    end

    constraints = JuMP.all_constraints(pm.model; include_variable_in_set_constraints=true)
    relaxation_penalty = JuMP.MOI.Utilities.PenaltyRelaxation(Dict(JuMP.index(c) => 5.0 for c in constraints))
    backend = JuMP.backend(pm.model)
    penalty_dict = JuMP.MOI.modify(backend, relaxation_penalty)

    for (pg, sp) in zip(pgs, pg_sps)
        constraint_pbal_sp(pm, pg, sp)
    end
    pm.data["penalty_dict"] = penalty_dict
    pm.data["constraint_dict"] = constraint_dict
end

"a toy example of how to model with multi-networks"
function solve_mn_opf(file, model_type::Type, optimizer; kwargs...)
    return solve_model(file, model_type, optimizer, build_mn_opf; multinetwork=true, kwargs...)
end

"""
    build_mn_opf(pm::AbstractPowerModel)
"""
function build_mn_opf(pm::AbstractPowerModel)
    for (n, network) in nws(pm)
        variable_bus_voltage(pm, nw=n)
        variable_gen_power(pm, nw=n)
        variable_branch_power(pm, nw=n)
        variable_dcline_power(pm, nw=n)

        constraint_model_voltage(pm, nw=n)

        for i in ids(pm, :ref_buses, nw=n)
            constraint_theta_ref(pm, i, nw=n)
        end

        for i in ids(pm, :bus, nw=n)
            constraint_power_balance(pm, i, nw=n)
        end

        for i in ids(pm, :branch, nw=n)
            constraint_ohms_yt_from(pm, i, nw=n)
            constraint_ohms_yt_to(pm, i, nw=n)

            constraint_voltage_angle_difference(pm, i, nw=n)

            constraint_thermal_limit_from(pm, i, nw=n)
            constraint_thermal_limit_to(pm, i, nw=n)
        end

        for i in ids(pm, :dcline, nw=n)
            constraint_dcline_power_losses(pm, i, nw=n)
        end
    end

    objective_min_fuel_and_flow_cost(pm)
end


"a toy example of how to model with multi-networks and storage"
function solve_mn_opf_strg(file, model_type::Type, optimizer; kwargs...)
    return solve_model(file, model_type, optimizer, build_mn_opf_strg; multinetwork=true, kwargs...)
end

"""
    build_mn_opf_strg(pm::AbstractPowerModel)
"""
function build_mn_opf_strg(pm::AbstractPowerModel)
    for (n, network) in nws(pm)
        variable_bus_voltage(pm, nw=n)
        variable_gen_power(pm, nw=n)
        variable_storage_power_mi(pm, nw=n)
        variable_branch_power(pm, nw=n)
        variable_dcline_power(pm, nw=n)

        constraint_model_voltage(pm, nw=n)

        for i in ids(pm, :ref_buses, nw=n)
            constraint_theta_ref(pm, i, nw=n)
        end

        for i in ids(pm, :bus, nw=n)
            constraint_power_balance(pm, i, nw=n)
        end

        for i in ids(pm, :storage, nw=n)
            constraint_storage_complementarity_mi(pm, i, nw=n)
            constraint_storage_losses(pm, i, nw=n)
            constraint_storage_thermal_limit(pm, i, nw=n)
        end

        for i in ids(pm, :branch, nw=n)
            constraint_ohms_yt_from(pm, i, nw=n)
            constraint_ohms_yt_to(pm, i, nw=n)

            constraint_voltage_angle_difference(pm, i, nw=n)

            constraint_thermal_limit_from(pm, i, nw=n)
            constraint_thermal_limit_to(pm, i, nw=n)
        end

        for i in ids(pm, :dcline, nw=n)
            constraint_dcline_power_losses(pm, i, nw=n)
        end
    end

    network_ids = sort(collect(nw_ids(pm)))

    n_1 = network_ids[1]
    for i in ids(pm, :storage, nw=n_1)
        constraint_storage_state(pm, i, nw=n_1)
    end

    for n_2 in network_ids[2:end]
        for i in ids(pm, :storage, nw=n_2)
            constraint_storage_state(pm, i, n_1, n_2)
        end
        n_1 = n_2
    end

    objective_min_fuel_and_flow_cost(pm)
end




"""
Solves an opf using ptdfs with no explicit voltage or line flow variables.

This formulation is most often used when a small subset of the line flow
constraints are active in the data model.
"""
function solve_opf_ptdf(file, model_type::Type, optimizer; full_inverse=false, kwargs...)
    if !full_inverse
        return solve_model(file, model_type, optimizer, build_opf_ptdf; ref_extensions=[ref_add_connected_components!,ref_add_sm!], kwargs...)
    else
        return solve_model(file, model_type, optimizer, build_opf_ptdf; ref_extensions=[ref_add_connected_components!,ref_add_sm_inv!], kwargs...)
    end
end

function build_opf_ptdf(pm::AbstractPowerModel)
    Memento.error(_LOGGER, "build_opf_ptdf is only valid for DCPPowerModels")
end

"""
    build_opf_ptdf(pm::DCPPowerModel)
"""
function build_opf_ptdf(pm::DCPPowerModel)
    variable_gen_power(pm)

    for i in ids(pm, :bus)
        expression_bus_power_injection(pm, i)
    end

    objective_min_fuel_cost(pm)

    constraint_model_voltage(pm)

    # this constraint is implicit in this model
    #for i in ids(pm, :ref_buses)
    #    constraint_theta_ref(pm, i)
    #end

    for i in ids(pm, :components)
        constraint_network_power_balance(pm, i)
    end

    for (i, branch) in ref(pm, :branch)
        # requires optional vad parameters
        #constraint_voltage_angle_difference(pm, i)

        # only create these expressions if a line flow is specified
        if haskey(branch, "rate_a")
            expression_branch_power_ohms_yt_from_ptdf(pm, i)
            expression_branch_power_ohms_yt_to_ptdf(pm, i)
        end

        constraint_thermal_limit_from(pm, i)
        constraint_thermal_limit_to(pm, i)
    end
end


function ref_add_sm!(ref::Dict{Symbol, <:Any}, data::Dict{String, <:Any})
    apply_pm!(_ref_add_sm!, ref, data)
end


function _ref_add_sm!(ref::Dict{Symbol, <:Any}, data::Dict{String, <:Any})
    reference_bus(data) # throws an error if an incorrect number of reference buses are defined
    ref[:sm] = calc_susceptance_matrix(data)
end


function ref_add_sm_inv!(ref::Dict{Symbol, <:Any}, data::Dict{String, <:Any})
    apply_pm!(_ref_add_sm_inv!, ref, data)
end


function _ref_add_sm_inv!(ref::Dict{Symbol, <:Any}, data::Dict{String, <:Any})
    ref[:sm] = calc_susceptance_matrix_inv(data)
end
