using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Pkg.instantiate
using Revise
using PowerModels
include("../config.jl")
include("./find_nearest_gens.jl")
push!(LOAD_PATH, DATA_PATH)
using Infiltrator
Infiltrator.toggle_async_check(false)
using DataFrames
using OrderedCollections
using DataStructures
using JSON
using XLSX
using Ipopt
using Infiltrator
using JuMP
# using Plots
using LinearAlgebra
using Distributions
using Random
using Graphs
Random.seed!(2)

function _determine_acpf_feasibility(test_case, viol_dict; epsilon=1e-5)
    total_violations, total_vm_violations, total_q_violations, total_branch_violations = 0, 0, 0, 0
    # Check flow limits 
    flows = PowerModels.calc_branch_flow_ac(test_case)["branch"]
    for (ind, flow) in pairs(flows)
        viol_dict["lf_viol_$ind"] =   Int((flow["pf"]^2  + flow["qf"]^2 <= test_case["branch"][ind]["rate_a"]^2 + epsilon) && 
                                        (flow["pt"]^2  + flow["qt"]^2 <= test_case["branch"][ind]["rate_a"]^2 + epsilon))
        total_violations += 1 - viol_dict["lf_viol_$ind"]
        total_branch_violations += 1 - viol_dict["lf_viol_$ind"]
    end

    # Check angle limits
    for (ind, branch) in pairs(test_case["branch"])
        viol_dict["la_viol_$ind"] =  Int((abs(test_case["bus"][string(branch["f_bus"])]["va"] - 
                                        test_case["bus"][string(branch["t_bus"])]["va"]) <= branch["angmax"] + epsilon) && 
                                        (abs(test_case["bus"][string(branch["f_bus"])]["va"] - 
                                        test_case["bus"][string(branch["t_bus"])]["va"]) >= branch["angmin"] - epsilon))
        total_violations += 1 - viol_dict["la_viol_$ind"]
        total_branch_violations += 1 - viol_dict["la_viol_$ind"]
    end

    # Voltage magnitude limits
    for (ind, bus) in pairs(test_case["bus"])
        viol_dict["vmviol_$ind"] =  Int(bus["vm"] <= bus["vmax"] + epsilon &&
                                 bus["vm"] + epsilon >= bus["vmin"])
        total_violations += 1 - viol_dict["vmviol_$ind"]
        total_vm_violations += 1 - viol_dict["vmviol_$ind"]
    end 

    # Reactive power limits
    for (ind, gen) in pairs(test_case["gen"])
        viol_dict["qviol_$ind"] =  Int(gen["qg"] <= gen["qmax"] + epsilon &&
                                gen["qg"] >= gen["qmin"] - epsilon)
        total_violations += 1 - viol_dict["qviol_$ind"]
        total_q_violations += 1 - viol_dict["qviol_$ind"]
    end

    # add other values 
    viol_dict["total_violations"] = total_violations 
    viol_dict["total_branch_violations"] = total_branch_violations 
    viol_dict["total_vm_violations"] = total_vm_violations 
    viol_dict["total_q_violations"] = total_q_violations 
    return viol_dict
end

function solution_feasibility(data, sol_dict, viol_dict; epsilon = 1e-5)
    # place values in data (un- PU)
    for (ind, val) in pairs(sol_dict["gen"])
        data["gen"][ind]["pg"] = val["pg"]
        data["gen"][ind]["qg"] = val["qg"]
    end
    for (ind, val) in pairs(sol_dict["bus"])
        data["bus"][ind]["va"] = val["va"]
        data["bus"][ind]["vm"] = val["vm"]
    end
    # determine feasibility 
    viol_dict =  _determine_acpf_feasibility(data, viol_dict; epsilon = epsilon)
    return  viol_dict
end

"""
`rand(Uniform(a, b))` errors when `a == b` (e.g. a delta/pct of 0 collapses
the range to a single point). Return that point directly instead of drawing.
"""
_rand_uniform(a, b) = a == b ? a : rand(Uniform(a, b))

"""
Draw a perturbed load (and, if `qd_loads` is given, reactive load) that
satisfies `_verify_loads_`, retrying up to `max_attempts` times. Iterative
rather than recursive on purpose: a recursive "retry on failure" (as this
used to be) has no bound on stack depth, and for a case/config combination
where no draw is ever feasible (e.g. `max_pd` too tight at some bus for the
configured `delta`/`jitter_delta` range) it eventually crashes with a
`StackOverflowError` instead of failing with a clear, diagnosable error --
or, worse, appears to just hang indefinitely accumulating stack frames.

Both `delta` (the global scaling range) and `jitter_delta` (the per-load
range) shrink on every retry -- linearly, from the full amount on attempt 1
down to `1/max_attempts` of it on the last attempt -- so a case that's only
occasionally pushed infeasible by an aggressive draw becomes more likely to
succeed the more it retries, rather than re-rolling the same range forever.
This can't rescue a case/bus that's already infeasible at zero perturbation
(shrinking `delta` toward 0 makes `global_scale` approach 1, i.e. the
nominal, unperturbed load -- it never pushes below what a smaller `delta`
already allows), so that failure mode still surfaces as the `error` below,
just after `max_attempts` tries instead of 1000.
"""
function perturb_load!(test_case, loads, delta, max_pd; qd_loads = nothing, jitter_delta = 0.05, max_attempts = 20)
    # get max and min total generation
    max_gen = sum([gen["pmax"] for gen in values(test_case["gen"])])
    min_gen = sum([gen["pmin"] for gen in values(test_case["gen"])])
    max_qg = qd_loads === nothing ? nothing : sum([gen["qmax"] for gen in values(test_case["gen"])])
    min_qg = qd_loads === nothing ? nothing : sum([gen["qmin"] for gen in values(test_case["gen"])])

    for attempt in 1:max_attempts
        # shrink the perturbation range on every retry, from the full amount
        # (attempt 1) down to 1/max_attempts of it (the last attempt)
        shrink = 1 - (attempt - 1) / max_attempts
        attempt_delta = delta * shrink
        attempt_jitter_delta = jitter_delta * shrink

        # single global scaling factor applied to every load (+-attempt_delta)
        global_scale = _rand_uniform(max(1-attempt_delta, 0), 1+attempt_delta)
        # perturb load: global scale * per-load jitter (+-attempt_jitter_delta)
        for load in values(test_case["load"])
            jitter = _rand_uniform(max(1-attempt_jitter_delta, 0), 1+attempt_jitter_delta)
            load["pd"] = global_scale * jitter * load["og_pd"]
            loads[load["index"]] = load["pd"]
        end
        if qd_loads !== nothing
            # perturb qd load
            for load in values(test_case["load"])
                jitter = _rand_uniform(max(1-attempt_jitter_delta, 0), 1+attempt_jitter_delta)
                load["qd"] = global_scale * jitter * load["og_qd"]
                qd_loads[load["index"]] = load["qd"]
            end
        end

        feasible = qd_loads === nothing ?
            _verify_loads_(test_case, loads, max_gen, min_gen, max_pd) :
            _verify_loads_(test_case, loads, max_gen, min_gen, max_pd; qd_loads = qd_loads, max_qg = max_qg, min_qg = min_qg)
        if feasible
            return test_case, loads, qd_loads
        end
    end
    error("perturb_load!: no feasible load draw found in $max_attempts attempts (starting delta=$delta, jitter_delta=$jitter_delta, shrinking toward the nominal load each retry) -- max_pd or generation bounds may be too tight for this case/config even near zero perturbation")
end

# -----------------------------------------------------------------------------
# Perturbation config
#
# Every percentage/fraction used by the perturbation functions below lives in
# `perturbation_configs/default_pert.json` (next to this file, path stored in
# PERTURBATION_CONFIGS_DIR/DEFAULT_PERTURBATION_CONFIG_PATH). Call
# `load_perturbation_config()` to get the defaults, optionally passing the
# full path to a second JSON file whose values override the defaults (only
# the keys you supply are overridden - anything you omit falls back to the
# default). The resulting nested Dict can be splatted into the perturbation
# functions, e.g.:
#
#   cfg = load_perturbation_config(joinpath(PERTURBATION_CONFIGS_DIR, "my_overrides.json"))
#   shuffle_gencost!(test_case; shuffle_fraction = cfg["gencost"]["shuffle_fraction"])
#
# `generate_data`'s `pert_config_path` kwarg takes just a filename (or an
# absolute path) and resolves it against PERTURBATION_CONFIGS_DIR itself, so
# it works no matter what the shell's current working directory is.
# -----------------------------------------------------------------------------

const PERTURBATION_CONFIGS_DIR = joinpath(@__DIR__, "perturbation_configs")
const DEFAULT_PERTURBATION_CONFIG_PATH = joinpath(PERTURBATION_CONFIGS_DIR, "default_pert.json")

function load_perturbation_config(override_path::Union{Nothing, AbstractString} = nothing;
                                   default_path::AbstractString = DEFAULT_PERTURBATION_CONFIG_PATH)
    config = JSON.parsefile(default_path; dicttype = Dict{String, Any})
    if override_path !== nothing
        overrides = JSON.parsefile(override_path; dicttype = Dict{String, Any})
        _merge_perturbation_config!(config, overrides)
    end
    return config
end

function _merge_perturbation_config!(base::Dict, overrides::Dict)
    for (k, v) in overrides
        if haskey(base, k) && isa(base[k], Dict) && isa(v, Dict)
            _merge_perturbation_config!(base[k], v)
        else
            base[k] = v
        end
    end
    return base
end

# -----------------------------------------------------------------------------
# Perturbation functions
#
# Each function mutates `test_case` in place (hence the "!") and also returns
# it, so calls can be chained, e.g. `test_case |> shuffle_gencost! |> perturb_gen!`.
# -----------------------------------------------------------------------------

"""
Shuffle the generator cost coefficient vectors ("cost") among a random subset
of generators. `shuffle_fraction` (0-1) controls what fraction of generators
participate in the shuffle.
"""
function shuffle_gencost!(test_case; shuffle_fraction = 0.20)
    gen_ids = collect(keys(test_case["gen"]))
    num_to_shuffle = round(Int, shuffle_fraction * length(gen_ids))
    if num_to_shuffle < 2
        return test_case
    end
    selected = shuffle(gen_ids)[1:num_to_shuffle]
    costs = [deepcopy(test_case["gen"][g]["cost"]) for g in selected]
    shuffled_costs = shuffle(costs)
    for (g, cost) in zip(selected, shuffled_costs)
        test_case["gen"][g]["cost"] = cost
    end
    return test_case
end

"""
Squeeze the [pmin, pmax] range for a random subset of generators.
`gen_fraction` (0-1) controls how many generators are affected; for each
selected generator a squeeze percentage is drawn uniformly from
[0, max_squeeze_pct] and applied symmetrically to pmin/pmax.
"""
function perturb_gen!(test_case; gen_fraction = 0.50, max_squeeze_pct = 0.20)
    gen_ids = collect(keys(test_case["gen"]))
    num_to_squeeze = round(Int, gen_fraction * length(gen_ids))
    selected = shuffle(gen_ids)[1:num_to_squeeze]
    for g in selected
        gen = test_case["gen"][g]
        squeeze_pct = _rand_uniform(0, max_squeeze_pct)
        prange = gen["pmax"] - gen["pmin"]
        gen["pmin"] = gen["pmin"] + (squeeze_pct / 2) * prange
        gen["pmax"] = gen["pmax"] - (squeeze_pct / 2) * prange
    end
    return test_case
end

"""
Squeeze the [vmin, vmax] range for a random subset of buses.
`bus_fraction` (0-1) controls how many buses are affected; for each selected
bus a squeeze percentage is drawn uniformly from [0, max_squeeze_pct] and
applied symmetrically to vmin/vmax.
"""
function vsqueeze!(test_case; bus_fraction = 0.30, max_squeeze_pct = 0.20)
    bus_ids = collect(keys(test_case["bus"]))
    num_to_squeeze = round(Int, bus_fraction * length(bus_ids))
    selected = shuffle(bus_ids)[1:num_to_squeeze]
    for b in selected
        bus = test_case["bus"][b]
        squeeze_pct = _rand_uniform(0, max_squeeze_pct)
        vrange = bus["vmax"] - bus["vmin"]
        bus["vmin"] = bus["vmin"] + (squeeze_pct / 2) * vrange
        bus["vmax"] = bus["vmax"] - (squeeze_pct / 2) * vrange
    end
    return test_case
end

"""
Squeeze the [angmin, angmax] voltage angle difference range for a random
subset of lines. `line_fraction` (0-1) controls how many branches are
affected; for each selected branch a squeeze percentage is drawn uniformly
from [0, max_squeeze_pct] and applied symmetrically to angmin/angmax.
"""
function vasqueeze!(test_case; line_fraction = 0.30, max_squeeze_pct = 0.10)
    branch_ids = collect(keys(test_case["branch"]))
    num_to_squeeze = round(Int, line_fraction * length(branch_ids))
    selected = shuffle(branch_ids)[1:num_to_squeeze]
    for l in selected
        branch = test_case["branch"][l]
        squeeze_pct = _rand_uniform(0, max_squeeze_pct)
        arange = branch["angmax"] - branch["angmin"]
        branch["angmin"] = branch["angmin"] + (squeeze_pct / 2) * arange
        branch["angmax"] = branch["angmax"] - (squeeze_pct / 2) * arange
    end
    return test_case
end

"""
Squeeze the thermal rating ("rate_a") of a random subset of lines.
`line_fraction` (0-1) controls how many branches are affected; for each
selected branch a squeeze percentage is drawn uniformly from
[0, max_squeeze_pct] and rate_a is scaled down by (1 - squeeze_pct).
"""
function thermal_squeeze!(test_case; line_fraction = 0.20, max_squeeze_pct = 0.20)
    branch_ids = collect(keys(test_case["branch"]))
    num_to_squeeze = round(Int, line_fraction * length(branch_ids))
    selected = shuffle(branch_ids)[1:num_to_squeeze]
    for l in selected
        branch = test_case["branch"][l]
        squeeze_pct = _rand_uniform(0, max_squeeze_pct)
        branch["rate_a"] = branch["rate_a"] * (1 - squeeze_pct)
    end
    return test_case
end

"""
Perturb the shunt susceptance ("bs") of a random subset of shunts by
+-bs_delta (multiplicative, e.g. bs_delta = 0.10 means +-10%).
`shunt_fraction` (0-1) controls what fraction of shunts are perturbed.
"""
function perturb_sus!(test_case; bs_delta = 0.10, shunt_fraction = 0.20)
    shunt_ids = collect(keys(get(test_case, "shunt", Dict())))
    num_to_perturb = round(Int, shunt_fraction * length(shunt_ids))
    selected = shuffle(shunt_ids)[1:num_to_perturb]
    for s in selected
        shunt = test_case["shunt"][s]
        pert = _rand_uniform(-bs_delta, bs_delta)
        shunt["bs"] = shunt["bs"] * (1 + pert)
    end
    return test_case
end

"""
Generator and line outages (`pert_genstatus!`/`pert_linestatus!`) are only
applied to networks larger than this many buses. Small test cases (e.g.
case9/case14) are prone to having every line be a bridge and every generator
be load-critical, so a single outage often leaves the network unable to
serve its load at all; larger networks have enough redundancy for outages to
be a meaningful, usually-still-solvable perturbation.
"""
const OUTAGE_MIN_BUSES = 300

"""
Turn OFF (`gen_status = 0`) `num_gens` randomly-selected, currently in-service
generators (an outright count, not a fraction). The slack-bus generator is
never turned off, since a network with no reference bus can't be solved.
Meant to be called before DC-OPF, so both the DC- and AC-feasibility checks
run against the reduced generator fleet. No-op on networks with
`OUTAGE_MIN_BUSES` buses or fewer.

Runs `PowerModels.correct_bus_types!` afterward: a PV bus whose only
generator just got turned off would otherwise be left with `bus_type == 2`
but zero active generators, which the AC-PF solver doesn't handle gracefully
(it assumes every PV bus has an active generator, and crashes with a
`KeyError` rather than failing gracefully otherwise); `correct_bus_types!`
demotes it to a PQ bus (`bus_type = 1`), same as PowerModels does for any
other topology change.
"""
function pert_genstatus!(test_case; num_gens::Integer = 0)
    num_gens <= 0 && return test_case
    length(test_case["bus"]) <= OUTAGE_MIN_BUSES && return test_case
    slack_bus = [bus["bus_i"] for bus in values(test_case["bus"]) if bus["bus_type"] == 3][1]
    eligible_ids = [ind for (ind, gen) in test_case["gen"]
                    if get(gen, "gen_status", 1) != 0 && gen["gen_bus"] != slack_bus]
    num_to_turn_off = min(num_gens, length(eligible_ids))
    selected = shuffle(eligible_ids)[1:num_to_turn_off]
    for g in selected
        test_case["gen"][g]["gen_status"] = 0
    end
    if !isempty(selected)
        PowerModels.correct_bus_types!(test_case)
    end
    return test_case
end

"""
Turn OFF (`br_status = 0`) `num_lines` randomly-selected, currently in-service
branches (an outright count, not a fraction). Meant to be called after
DC-OPF but before the AC-PF solve, modeling a line outage that happens after
dispatch (e.g. an N-1 contingency) rather than one DC-OPF dispatched around.

Candidates are drawn in random order and accepted one at a time: a candidate
is only turned off if doing so leaves its two endpoint buses still connected
to each other by some other path (checked directly against the graph as it
stands *after* every previously-accepted removal, via `Graphs.has_path`) --
if not, that candidate is skipped (left in service) and the next random
candidate is tried instead, until `num_lines` have been turned off or no
candidates remain. This guarantees no bus ever ends up islanded, even when
`num_lines > 1` and no single one of the removed lines would have been
unsafe on its own but the *combination* would island a bus. An islanded bus
isn't a case the downstream AC-PF solver handles gracefully -- it crashes
with a low-level error (summing over an empty neighbor set) rather than
reporting infeasibility the way the rest of this pipeline expects.

No-op on networks with `OUTAGE_MIN_BUSES` buses or fewer.
"""
function pert_linestatus!(test_case; num_lines::Integer = 0)
    num_lines <= 0 && return test_case
    length(test_case["bus"]) <= OUTAGE_MIN_BUSES && return test_case
    in_service = [(ind, branch) for (ind, branch) in test_case["branch"] if get(branch, "br_status", 1) != 0]
    isempty(in_service) && return test_case

    # `Graphs.SimpleGraph` has no concept of parallel edges, so two branches
    # on the same bus pair collapse into a single graph edge; `pair_remaining`
    # tracks the true (still in-service) count per pair so a redundant
    # parallel branch can be removed freely without ever touching the graph
    bus_ids = sort(unique(vcat([b["f_bus"] for (_, b) in in_service], [b["t_bus"] for (_, b) in in_service])))
    bus_to_vertex = Dict(bus => v for (v, bus) in enumerate(bus_ids))
    g = Graphs.SimpleGraph(length(bus_ids))
    branch_pair = Dict{String, Tuple{Int,Int}}()
    pair_remaining = Dict{Tuple{Int,Int}, Int}()
    for (ind, branch) in in_service
        u, v = bus_to_vertex[branch["f_bus"]], bus_to_vertex[branch["t_bus"]]
        pair = u < v ? (u, v) : (v, u)
        Graphs.add_edge!(g, pair[1], pair[2])
        branch_pair[ind] = pair
        pair_remaining[pair] = get(pair_remaining, pair, 0) + 1
    end

    turned_off = String[]
    for ind in shuffle([ind for (ind, _) in in_service])
        length(turned_off) >= num_lines && break
        pair = branch_pair[ind]
        if pair_remaining[pair] > 1
            # a parallel branch on this same bus pair is still in service
            pair_remaining[pair] -= 1
            push!(turned_off, ind)
            continue
        end
        u, v = pair
        Graphs.rem_edge!(g, u, v)
        if Graphs.has_path(g, u, v)
            # some other route still connects u and v
            pair_remaining[pair] -= 1
            push!(turned_off, ind)
        else
            Graphs.add_edge!(g, u, v)   # revert; this line would island a bus, try another
        end
    end

    for ind in turned_off
        test_case["branch"][ind]["br_status"] = 0
    end
    return test_case
end

# Real OLTC/phase-shifter tap changers move in fixed discrete steps rather
# than a continuous range, so perturbations are drawn from a discrete grid
# of step multiples (..., -2, -1, 0, 1, 2, ...) instead of `rand(Uniform(...))`.
const TAP_STEP_PCT = 0.006   # 0.6% per discrete tap-changer step
const SHIFT_STEP_DEG = 2.0   # degrees per discrete phase-shifter step
const SHIFT_STEP_RAD = deg2rad(SHIFT_STEP_DEG)

"""
Classify the branches of `test_case` that Matpower encodes as transformers
(`transformer == true`) into tap-changing and phase-shifting sets. Matpower
only distinguishes a transformer from an ordinary line via `tap != 1.0` (an
off-nominal turns ratio) or `shift` differing from "no shift" - and some case
files encode "no shift" as a full rotation (±360 degrees, i.e. ±2π radians)
rather than exactly 0, so that's checked too. A transformer can be both a
tap-changer and a phase-shifter at once, so the two returned ID vectors may
overlap. Call this once on the base case, before the generation loop, so
every sample perturbs the same fixed sets of eligible branches rather than
re-deriving (and potentially drifting) them on each perturbed copy.
"""
function classify_transformers(test_case; angle_eps = 1e-6)
    full_rotation = 2 * pi  # shift is stored in radians; 360 degrees == 2*pi
    tap_changing_ids = String[]
    phase_shifting_ids = String[]
    for (ind, branch) in test_case["branch"]
        get(branch, "transformer", false) || continue
        if branch["tap"] != 1.0
            push!(tap_changing_ids, ind)
        end
        shift_mod = mod(branch["shift"], full_rotation)
        is_no_shift = isapprox(shift_mod, 0.0; atol = angle_eps) ||
                      isapprox(shift_mod, full_rotation; atol = angle_eps)
        if !is_no_shift
            push!(phase_shifting_ids, ind)
        end
    end
    return tap_changing_ids, phase_shifting_ids
end

"""
Perturb the tap ratio of a random subset of `eligible_ids` (the tap-changing
transformer branch IDs returned by `classify_transformers`) by a random
integer multiple (in `-max_steps:max_steps`) of `step_pct` (multiplicative),
mimicking a discrete mechanical tap changer (default step: 0.6%, i.e. a ±10%
range over 33 physical positions). `tap_fraction` (0-1) controls what
fraction of `eligible_ids` are perturbed; `max_steps` controls how many steps
(in either direction) a selected transformer may move.
"""
function perturb_tap!(test_case, eligible_ids; max_steps::Integer = 10, tap_fraction = 0.20, step_pct = TAP_STEP_PCT)
    num_to_perturb = round(Int, tap_fraction * length(eligible_ids))
    selected = shuffle(eligible_ids)[1:num_to_perturb]
    for l in selected
        branch = test_case["branch"][l]
        step = rand(-max_steps:max_steps)
        branch["tap"] = branch["tap"] * (1 + step * step_pct)
    end
    return test_case
end

"""
Perturb the phase shift of a random subset of `eligible_ids` (the
phase-shifting transformer branch IDs returned by `classify_transformers`) by
a random integer multiple (in `-max_steps:max_steps`) of `step_rad` (additive
- phase shift is an angle, not a ratio), mimicking a discrete mechanical tap
changer (default step: 2 degrees). `test_case["branch"][...]["shift"]` is
stored in radians by PowerModels, so `step_rad` defaults to
`SHIFT_STEP_RAD = deg2rad(SHIFT_STEP_DEG)`. `shift_fraction` (0-1) controls
what fraction of `eligible_ids` are perturbed; `max_steps` controls how many
steps (in either direction) a selected transformer may move.
"""
function perturb_shift!(test_case, eligible_ids; max_steps::Integer = 10, shift_fraction = 0.20, step_rad = SHIFT_STEP_RAD)
    num_to_perturb = round(Int, shift_fraction * length(eligible_ids))
    selected = shuffle(eligible_ids)[1:num_to_perturb]
    for l in selected
        branch = test_case["branch"][l]
        step = rand(-max_steps:max_steps)
        branch["shift"] = branch["shift"] + step * step_rad
    end
    return test_case
end

"""
Add tap/shift tuning bounds and discrete setpoint grids to the branches in
`tap_ids`/`shift_ids` (the tap-changing/phase-shifting branch IDs from
`classify_transformers`), and shunt susceptance bounds to every shunt, for
use by `PowerModels.build_dc_ac_device_pf`. Every flagged branch shares the
same physical tap-changer/phase-shifter setpoint grid, centered on nominal
(tap = 1, shift = 0) rather than that branch's own tap/shift -- e.g.
`tap_setpoints = [0.90, 0.9075, 0.915, ..., 1.095]`,
`shift_setpoints = [-0.3, -0.25, ..., 0.3]` -- since real tap-changer
positions are fixed steps around nominal, the same for every transformer of
a given design, regardless of where a particular one currently sits. Shunt
`bmin`/`bmax` stay relative to each shunt's own `bs` (susceptance has no
equivalent fixed physical grid here); `minmax(...)` guards against the
bounds coming out reversed when `bs` is negative. Mutates and returns
`test_case`.
"""
function prepare_transformer_adjustments(test_case, tap_ids, shift_ids)
    tap_range = 0.1
    shift_range = 0.3
    shunt_range = 0.4
    tap_step = 0.0075
    shift_step = 0.05

    tap_setpoints = round.(1 .+ collect(range(-tap_range, tap_range; step = tap_step)); digits = 4)
    shift_setpoints = round.(collect(range(-shift_range, shift_range; step = shift_step)); digits = 4)
    tmin, tmax = extrema(tap_setpoints)
    smin, smax = extrema(shift_setpoints)

    for l in tap_ids
        branch = test_case["branch"][l]
        branch["is_tap"] = true
        branch["tmin"], branch["tmax"] = tmin, tmax
        branch["tap_setpoints"] = copy(tap_setpoints)
    end
    for l in shift_ids
        branch = test_case["branch"][l]
        branch["is_shift"] = true
        branch["smin"], branch["smax"] = smin, smax
        branch["shift_setpoints"] = copy(shift_setpoints)
    end
    for shunt in values(test_case["shunt"])
        shunt["bmin"], shunt["bmax"] = minmax(shunt["bs"]*(1 - shunt_range), shunt["bs"]*(1 + shunt_range))
    end

    return test_case
end

"""
    apply_solution!(test_case, result)

Writes a `PowerModels` OPF/PF `result`'s solution back into `test_case`: bus
`vm`/`va`, gen `pg`/`qg` (with `vg` updated to the solved `vm` at that
generator's bus -- the AC-feasible point a setpoint-distance objective like
`build_dc_ac_pf`/`build_dc_ac_device_pf` converged to), and, for any branch
carrying a solved `tap`/`shift` (i.e. one `build_dc_ac_device_pf` treated as
tunable, per `prepare_transformer_adjustments`), that branch's `tap`/`shift`
snapped to the nearest entry in its `tap_setpoints`/`shift_setpoints` grid.
Reusable across any solve whose solution has this "bus"/"gen"/"branch" shape
-- the tap/shift snap is simply skipped for a branch/result that doesn't
carry it. Iterates over `solution`'s components, not `test_case`'s: PowerModels
only includes *active* buses/gens/branches in a solution (an out-of-service
generator, e.g. one `pert_genstatus!` turned off, or an islanded bus, never
gets a variable and so is simply absent from `solution`, even though it's
still listed in `test_case`) -- so a component missing from `solution` is
left as-is in `test_case` rather than raising a `KeyError`. Mutates and
returns `test_case`.
"""
function apply_solution!(test_case, result)
    solution = result["solution"]

    for (i, val) in solution["bus"]
        bus = test_case["bus"][i]
        bus["vm"] = val["vm"]
        bus["va"] = val["va"]
    end

    for (i, val) in solution["gen"]
        gen = test_case["gen"][i]
        gen["pg"] = val["pg"]
        gen["qg"] = val["qg"]
        gen["vg"] = solution["bus"][string(gen["gen_bus"])]["vm"]
    end

    for (i, val) in solution["branch"]
        branch = test_case["branch"][i]
        if get(branch, "is_tap", false) && haskey(val, "tap")
            branch["tap"] = argmin(sp -> abs(sp - val["tap"]), branch["tap_setpoints"])
        end
        if get(branch, "is_shift", false) && haskey(val, "shift")
            branch["shift"] = argmin(sp -> abs(sp - val["shift"]), branch["shift_setpoints"])
        end
    end

    return test_case
end

"""
    solve_dc_ac_pf!(test_case, optimizer; device::Bool = false)

Solves for the AC-feasible operating point closest to the generator voltage
setpoints in `test_case` (`PowerModels.solve_dc_ac_pf`), writing the solution
back into `test_case` via `apply_solution!`.

If `device = true`, first solves the same problem but also letting
transformer tap/shift (and shunt susceptance) move continuously
(`PowerModels.solve_dc_ac_device_pf`, after flagging the tunable branches via
`classify_transformers` + `prepare_transformer_adjustments`), and applies
that solution -- which snaps the solved tap/shift to the nearest entry in
their discrete setpoint grid. Only then is the plain (non-device) problem
solved and applied: `build_dc_ac_device_pf` optimizes tap/shift as
continuous and knows nothing about the discrete grid, so this final resolve
-- now against the branches' snapped, fixed tap/shift -- is what actually
finds the best AC-feasible point given those discretized values.

Returns `(test_case, result)` -- `result` is the *final* (non-device) solve's
full `PowerModels` result dict (`"termination_status"`, `"objective"`,
`"solve_time"`, ...), since that's the solve whose solution ends up in
`test_case`.
"""
function solve_dc_ac_pf!(test_case, optimizer; device::Bool = false)
    if device
        tap_ids, shift_ids = classify_transformers(test_case)
        prepare_transformer_adjustments(test_case, tap_ids, shift_ids)
        device_result = PowerModels.solve_dc_ac_device_pf(test_case, optimizer)
        apply_solution!(test_case, device_result)
        # `build_dc_ac_device_pf`, like `build_dc_ac_pf`, stashes its
        # soft-bound relaxation info in `pm.data["soft_bound_penalty_dict"]`
        # -- and `pm.data` *is* `test_case` (PowerModels doesn't copy it), so
        # this leaks a `Dict{MOI.ConstraintIndex, ...}` into test_case that
        # isn't JSON-serializable. Drop it now so it doesn't carry into the
        # next solve's data or into a caller that persists test_case.
        delete!(test_case, "soft_bound_penalty_dict")
    end
    result = PowerModels.solve_dc_ac_pf(test_case, optimizer)
    apply_solution!(test_case, result)
    delete!(test_case, "soft_bound_penalty_dict")
    return test_case, result
end

"""
    generate_optimal_dataset(dataset_path; device::Bool = false, log_level::String = "warn")

For every `<n>.json` case file directly inside `dataset_path/baseline_acpf`
(the raw, unsolved cases -- see `move_to_baseline_acpf`; same
`^\\d+\\.json\$` naming convention `store_datapoint!`/`generate_data` write),
parses it, solves it with `solve_dc_ac_pf!(...; device = device)`, and writes
the resulting test case to `<dataset_path>/dc_ac_device_pf/<n>.json` when
`device = true`, or `<dataset_path>/dc_ac_pf/<n>.json` otherwise -- a sibling
of `baseline_acpf`, not nested inside it -- always under the exact same
`<n>.json` filename the input case file had, so e.g. `1.json` in
`baseline_acpf/`, `dc_ac_pf/`, and `dc_ac_device_pf/` are all the same
underlying case, and a later comparison across those directories can just
pull matching filenames.

A case whose final solve doesn't reach `LOCALLY_SOLVED` is skipped (not
written) rather than persisting an unconverged operating point -- which means
that case's filename simply won't exist in `out_dir`, even though it does in
`baseline_acpf` (and possibly in the *other* device-flag's output directory,
if that one converged). Skipped filenames are printed (not just a count) so
that's visible before it surprises a later cross-directory pull.

Also writes `metadata.xlsx` into `out_dir`, one row per *attempted* case file
(including skipped ones -- `datapoint` is the only column guaranteed to line
up with what actually landed in `out_dir`) with:
  - `datapoint`: the case's numeric filename, e.g. `7` for `7.json`
  - `feasible`: `true` only if the final solve reached `LOCALLY_SOLVED` *and*
    the resulting point has zero real AC-PF violations per
    `_determine_acpf_feasibility` -- `solve_dc_ac_pf`/`solve_dc_ac_device_pf`
    keep qg/vm bounds *soft* (penalized, not enforced), so `LOCALLY_SOLVED`
    alone doesn't mean those bounds actually held
  - `time`: wall-clock seconds spent inside `solve_dc_ac_pf!` for that case
    -- for `device = true` this covers *both* optimization calls (the device
    solve and the final resolve), not just the last one
  - `objective`: the final (non-device) solve's objective value

Returns the output directory path.
"""
function generate_optimal_dataset(dataset_path; device::Bool = false, log_level::String = "error")
    PowerModels.logger_config!(log_level)
    # see the matching comment in generate_data: Ipopt's console output is a
    # separate channel from PowerModels' logger, controlled by its own
    # "print_level" option, not by `PowerModels.logger_config!`
    ipopt_print_level = log_level == "error" ? 0 : 1
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => ipopt_print_level)

    in_dir = joinpath(dataset_path, "baseline_acpf")
    out_dir = joinpath(dataset_path, device ? "dc_ac_device_pf" : "dc_ac_pf")
    mkpath(out_dir)

    case_files = filter(f -> occursin(r"^\d+\.json$", f), readdir(in_dir))

    metadata = DataFrame(datapoint = Int[], feasible = Bool[], time = Float64[], objective = Float64[])

    written, skipped = 0, String[]
    for (processed, f) in enumerate(case_files)
        datapoint = parse(Int, splitext(f)[1])
        test_case = PowerModels.parse_file(joinpath(in_dir, f))
        # write the solved case out under the same filename `f` it came in
        # under (see docstring) -- never renumbered/reindexed
        local result
        elapsed = @elapsed (test_case, result) = solve_dc_ac_pf!(test_case, ipopt; device = device)
        solved = result["termination_status"] == LOCALLY_SOLVED
        # `_determine_acpf_feasibility` checks the real (hard) AC-PF limits
        # against test_case's now-solved vm/va/pg/qg -- separate from
        # `solved`, since qg/vm bounds are only ever *soft* in these builds
        feasible = solved && _determine_acpf_feasibility(test_case, Dict())["total_violations"] == 0
        push!(metadata, (datapoint, feasible, elapsed, result["objective"]))

        if !solved
            push!(skipped, f)
        else
            # pretty-print (4-space indent) so the file is easy to read by eye; keeps
            # the same Inf/NaN -> "null" handling PowerModels.export_file uses so the
            # file still round-trips through PowerModels.parse_file
            JSON.json(joinpath(out_dir, f), test_case; pretty = 4, allownan = true, inf = "null", ninf = "null", nan = "null")
            written += 1
        end

        if processed % 50 == 0
            println("generate_optimal_dataset: processed $processed/$(length(case_files)) (written $written, skipped $(length(skipped))) -- $out_dir")
        end
    end

    metadata_path = joinpath(out_dir, "metadata.xlsx")
    isfile(metadata_path) && rm(metadata_path)
    XLSX.openxlsx(metadata_path, mode = "w") do xf
        sheet = XLSX.addsheet!(xf, "metadata")
        XLSX.writetable!(sheet, Tables.columntable(metadata))
    end

    println("generate_optimal_dataset: wrote $written cases to $out_dir")
    if !isempty(skipped)
        println("generate_optimal_dataset: skipped $(length(skipped)) unconverged case(s), missing from $out_dir: $(sort(skipped))")
    end
    return out_dir
end

function _verify_loads_(test_case, loads, max_gen, min_gen, max_pd; qd_loads = nothing, max_qg = nothing, min_qg = nothing)
    # check total load 
    if (sum(loads) > max_gen) || (sum(loads) < min_gen)
        return false
    end 
    if !isnothing(qd_loads)
        # check total qd
        if sum(qd_loads) > max_qg || sum(qd_loads) < min_qg
            return false
        end
    end
    # check load at each bus 
    for bus in values(test_case["bus"])
        if "bus_loads" ∉ keys(bus)
            continue
        end
        bus_load = sum([loads[index] for index in Set(bus["bus_loads"])])
        if bus_load > max_pd[string(bus["index"])]
            return false
        end
    end
    return true
end

"""
Per-branch `rate_a` (thermal rating) substituted into `max_pd` for a branch
that has no `"rate_a"` at all (Matpower's convention for "no rating
specified" -- see `prepare_test_case_perturbations`). Far above any real
branch rating (typically single/low-double-digit per-unit), so it behaves as
effectively unbounded without being a literal `Inf`.
"""
const UNRATED_BRANCH_MAX_PD = 1e6

function prepare_test_case_perturbations(test_case)
    for load in values(test_case["load"])
        load["og_pd"] = load["pd"]
        load["og_qd"] = load["qd"]
        if "bus_loads" ∉ keys(test_case["bus"][string(load["load_bus"])])
            test_case["bus"][string(load["load_bus"])]["bus_loads"] = [load["index"]]
        else 
            push!(test_case["bus"][string(load["load_bus"])]["bus_loads"], load["index"])
        end
    end
    max_pd = Dict{String, Float64}(bus => 0 for bus in keys(test_case["bus"]))
    for branch in values(test_case["branch"])
        # PowerModels deletes "rate_a" entirely (rather than storing 0) for a
        # branch whose Matpower RATE_A was 0, meaning "no thermal rating
        # specified" -- treat that as effectively unbounded, not as
        # contributing nothing: a bus with even one unrated incident branch
        # has no meaningful cap from this heuristic. A large finite sentinel
        # (rather than Inf) keeps max_pd an ordinary comparable/serializable
        # Float64 everywhere else it's used.
        rate_a = get(branch, "rate_a", UNRATED_BRANCH_MAX_PD)
        max_pd[string(branch["f_bus"])] = max_pd[string(branch["f_bus"])] + rate_a
        max_pd[string(branch["t_bus"])] = max_pd[string(branch["t_bus"])] + rate_a
    end
    return test_case, max_pd
end

function prepare_test_case(test_case, case_name, file_pth; solve_acpf = false)
    # set generator vm bounds so they're not immediately violated
    for (gen_ind, gen) in pairs(test_case["gen"])
        gen_bus = gen["gen_bus"]
        test_case["bus"][string(gen_bus)]["vmax"] = max(gen["vg"], test_case["bus"][string(gen_bus)]["vmax"])
    end
    # update line limits
    if isfile(joinpath(DATA_PATH, "test_cases/network_info/$case_name/pg_line_limits.txt"))
        line_lim_pth = joinpath(DATA_PATH, "test_cases/network_info/$case_name/pg_line_limits.txt")
        line_limits = read(line_lim_pth, String)
        line_limits = split(line_limits, ' ')[1:end-1]
        for (i, line) in enumerate(line_limits)
            test_case["branch"][string(i)]["rate_a"] = parse(Int, line)
        end
    end
    if solve_acpf
        # find nearest gens 
        nearest_gens = find_nearest_generators_khop(file_pth)
        test_case["pv_pairs"] = nearest_gens
    end
    return test_case
end


function store_load!(df, test_case, counter)
    dp_dict = Dict()
    for (gen_ind, gen) in pairs(test_case["gen"])
        dp_dict["pg_$gen_ind"] = gen["pg"]
    end
    for (i, (load_ind, load)) in enumerate(pairs(test_case["load"]))
       dp_dict["pd_$load_ind"] = load["pd"]
       dp_dict["qd_$load_ind"] = load["qd"]
    end
    dp_dict["datapoint"] = counter
    push!(df, dp_dict)
end

function generate_solutions(case_name, delta, test_case, load_data, file_pth, run_dict; 
                            num_samples = 1000, write_out = true)
    PowerModels.logger_config!("warn")
    # prep structures to store outputs
    cols = [("datapoint", Int64), ("run_id", Int64),
            ("pf_type", String), ("obo", Int64),
            ("swap_technique", String), ("grainger", Int64),
            ("use_smw_warmstart", Int64), ("score_collateral_aware", Int64),
            ("time", Float64),
            ("swap_iters", Int64), ("jac_iters", Int64)]
    run_df = DataFrame([name => type[] for (name, type) in cols])
    cols = vcat(["datapoint" ,"run_id", "iter", "jac_iter", "final_iter"], 
                ["qg_$gen_ind" for gen_ind in keys(test_case["gen"])],
                ["pg_$gen_ind" for gen_ind in keys(test_case["gen"])], 
                ["va_$bus_ind" for bus_ind in keys(test_case["bus"])], 
                ["vm_$bus_ind" for bus_ind in keys(test_case["bus"])])
    soln_df = DataFrame([name => Float64[] for name in cols])
    cols = vcat(["datapoint", "run_id", "iter"], 
                ["total_violations", "total_vm_violations", "total_q_violations", "total_branch_violations"],
                ["qviol_$gen_ind" for gen_ind in keys(test_case["gen"])],
                ["vmviol_$bus_ind" for bus_ind in keys(test_case["bus"])],
                ["lf_viol_$branch_ind" for branch_ind in keys(test_case["branch"])],
                ["la_viol_$branch_ind" for branch_ind in keys(test_case["branch"])]
                )
    violations_df = DataFrame([name => Float64[] for name in cols])
    cols = vcat(["datapoint", "run_id", "iter"], 
            ["bt_$bus_ind" for bus_ind in keys(test_case["bus"])]
            )
    bi_df = DataFrame([name => Float64[] for name in cols])
    nearest_gens = find_nearest_generators_khop(file_pth)

    # iterate through acpf combos 
    run_id = 0
    for pf_type in run_dict["pf_types"]
        for obo in run_dict["obo"]
            if pf_type in ["baseline", "qlim"]
                run_id += 1
                run_flags = Dict("run_id" => run_id, "pf_type" => pf_type,
                            "obo" => obo, "swap_technique" => "none", "grainger" => 0,
                            "use_smw_warmstart" => 0, "score_collateral_aware" => 0)
                println("Running ID $run_id: pf_type = $pf_type, obo = $obo")
                run_pf!(test_case, load_data, run_flags, run_df, soln_df, violations_df, bi_df, num_samples)
            else
                # `sensitivity_score` supports two extra knobs that other techniques ignore.
                # Iterate over them only when relevant; defaults of [0] keep this loop the
                # same shape as before for "nearest_gen" and "qv_inv".
                smw_grid = get(run_dict, "use_smw_warmstart", [0])
                collat_grid = get(run_dict, "score_collateral_aware", [0])
                for swap_technique in run_dict["swap_techniques"]
                    for grainger in run_dict["grainger"]
                        smw_loop = swap_technique == "sensitivity_score" ? smw_grid : [0]
                        collat_loop = swap_technique == "sensitivity_score" ? collat_grid : [0]
                        for use_smw in smw_loop
                            for collateral in collat_loop
                                run_id += 1
                                run_flags = Dict("run_id" => run_id, "pf_type" => pf_type,
                                            "obo" => obo, "swap_technique" => swap_technique,
                                            "grainger" => grainger,
                                            "use_smw_warmstart" => use_smw,
                                            "score_collateral_aware" => collateral
                                            )
                                test_case["pv_pairs"] = deepcopy(nearest_gens)
                                println("Running ID $run_id: pf_type=$pf_type, obo=$obo, swap_technique=$swap_technique, grainger=$grainger, use_smw=$use_smw, collat=$collateral")
                                run_pf!(test_case, load_data, run_flags, run_df, soln_df, violations_df, bi_df, num_samples)
                            end
                        end
                    end
                end
            end
        end
    end
    if write_out
        # write out data 
        filename = joinpath(RESULTS_PATH, "$(case_name)/$delta.xlsx")
        dir_path = dirname(filename)
        mkpath(dir_path)
        if isfile(filename)
            rm(filename)
        end
        XLSX.openxlsx(filename, mode="w") do xf
            sheet1 = XLSX.addsheet!(xf, "run_info")
            XLSX.writetable!(sheet1, Tables.columntable(run_df))
            sheet2 = XLSX.addsheet!(xf, "solns")
            XLSX.writetable!(sheet2, Tables.columntable(soln_df))
            sheet3 = XLSX.addsheet!(xf, "bus_types")
            XLSX.writetable!(sheet3, Tables.columntable(bi_df))
            sheet4 = XLSX.addsheet!(xf, "violations")
            XLSX.writetable!(sheet4, Tables.columntable(violations_df))
        end
    end
end

function run_pf!(original_test_case, load_data, run_flags, run_df, soln_df, violations_df, bi_df, num_samples)
    for (i, loads) in enumerate(eachrow(load_data))
        if i > num_samples 
            break 
        end
        test_case = deepcopy(original_test_case)
        # add values to test case 
        for (load_ind, load) in pairs(test_case["load"])
            load["pd"] = loads["pd_$load_ind"]
            load["qd"] = loads["qd_$load_ind"]
        end
        for (gen_ind, gen) in pairs(test_case["gen"])
            gen["pg"] = loads["pg_$gen_ind"]
        end
        # run ac power flow
        results = nothing
        if run_flags["pf_type"] in ["qlim", "baseline"]
            results = PowerModels.compute_ac_pf(test_case, mapping = true,
                                enforce_q_lims = run_flags["pf_type"] == "qlim")
        else
            # Optional sensitivity_score-only knobs (defaults preserve existing behavior).
            use_smw  = Bool(get(run_flags, "use_smw_warmstart", 0))
            collateral = Bool(get(run_flags, "score_collateral_aware", 0))
            results = PowerModels.compute_ac_pf_mult_buses(test_case, mapping = true,
                                    swap_technique = run_flags["swap_technique"],
                                    obo = Bool(run_flags["obo"]), grainger = Bool(run_flags["grainger"]),
                                    use_smw_warmstart = use_smw,
                                    score_collateral_aware = collateral, debug = true)
        end
        # parse results 
        parse_results!(test_case, results, run_flags, loads["datapoint"],  run_df, soln_df, violations_df, bi_df)
    end
end

function parse_results!(test_case, results, run_flags, datapoint, 
                        run_df, soln_df, violations_df, bi_df)
    run_id = run_flags["run_id"]
    # parse solutions 
    am = results["pf_data"].am
    num_swaps = length(results["solution_history"])
    jacobians = parse_jacobian_history(results["jacobian_history"])
    num_jacobians = sum([length(jac) for jac in jacobians])
    # add to run_df 
    run_flags["time"] = results["solve_time"]
    run_flags["swap_iters"] = num_swaps
    run_flags["jac_iters"] = num_jacobians
    run_flags["datapoint"] = datapoint 
    push!(run_df, run_flags)
    for iter in 1:num_swaps
        soln = results["solution_history"][iter]
        bus_indices = results["prev_bus_indices"][iter]
        # add to solution df 
        sol_dict = Dict("run_id" => run_id, "datapoint" => datapoint, 
                        "iter" => iter, 
                        "jac_iter" => length(jacobians[iter]), "final_iter" => iter == num_swaps)
        for (bus_ind, bus) in pairs(soln["bus"])
            sol_dict["vm_$bus_ind"] = bus["vm"]
            sol_dict["va_$bus_ind"] = bus["va"]
        end
        for (gen_ind, gen) in pairs(soln["gen"])
            sol_dict["pg_$gen_ind"] = gen["pg"]
            sol_dict["qg_$gen_ind"] = gen["qg"]
        end
        push!(soln_df, sol_dict)
        # add to bus_indices df 
        bi_dict = Dict("run_id" => run_id, "datapoint" => datapoint, "iter" => iter)
        for bus_ind in keys(soln["bus"])
            bi = am.bus_to_idx[parse(Int, bus_ind)]
            bi_dict["bt_$bus_ind"] = bus_indices[bi]
        end
        push!(bi_df, bi_dict)
        # add to violations df 
        viol_dict = Dict("run_id" => run_id, "datapoint" => datapoint, "iter" => iter)
        viol_dict = solution_feasibility(test_case, soln, viol_dict)
        push!(violations_df, viol_dict)
    end
end

function parse_jacobian_history(jacobian_history; is_mat = true)
    iteration_lst = []
    curr_iter = []
    for jac in jacobian_history
        if jac == []
            push!(iteration_lst, curr_iter)
            curr_iter = []
            continue 
        end 
        if is_mat
            push!(curr_iter, Matrix(jac))
        else 
            push!(curr_iter, jac)
        end
    end 
    return iteration_lst
end

function generate_loads(test_case, num_points, delta, case_name; log_level::String = "warn")
    PowerModels.logger_config!(log_level)
    # see the matching comment in generate_data: Ipopt's console output is a
    # separate channel from PowerModels' logger, controlled by its own
    # "print_level" option, not by `PowerModels.logger_config!`
    ipopt_print_level = log_level == "error" ? 0 : 1
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => ipopt_print_level)
    test_case, max_pd = prepare_test_case_perturbations(test_case)
    cols = vcat(["pd_$load_ind" for load_ind in keys(test_case["load"])],
                ["qd_$load_ind" for load_ind in keys(test_case["load"])],
                ["pg_$gen_ind" for gen_ind in keys(test_case["gen"])], ["datapoint"]
            )
    data = DataFrame([name => Float64[] for name in cols])
    counter = 0
    infeas_acopf = 0
    sorted_pairs = sort(collect(test_case["bus"]); by = x -> parse(Int, x.first))
    map_to_bus = [parse(Int64, i) for (i, bus) in sorted_pairs if bus["bus_type"] == 1]
    while counter < num_points
        if counter % 5 == 0
            println("counter = $counter")
        end
        # perturb test case load 
        loads = zeros(length(test_case["load"]))
        qd_loads = zeros(length(test_case["load"]))
        test_case, loads, qd_loads = perturb_load!(test_case, loads, delta, max_pd, qd_loads=qd_loads)
        # verify test case is dcopf-feasible 
        model_dc = PowerModels.solve_dc_opf(test_case, ipopt)
        if model_dc["termination_status"] != LOCALLY_SOLVED
            continue
        end 
        # verify test case is acopf-feasible (to guarantee feasible adjustment later)
        model_ac = PowerModels.solve_ac_opf(test_case, ipopt)
        if model_ac["termination_status"] != LOCALLY_SOLVED
            infeas_acopf += 1
            continue
        end 
        # place dc gen setpoints into model
        for (ind, val) in model_dc["solution"]["gen"]
            test_case["gen"][ind]["pg"] = val["pg"]
        end
        store_load!(data, test_case, counter)
        counter += 1
    end
    # store output
    filename = joinpath(TESTCASE_PATH, "data/$(case_name)/loads/$delta.xlsx")
    dir_path = dirname(filename)
    mkpath(dir_path)
    if isfile(filename)
        rm(filename)
    end
    XLSX.openxlsx(filename, mode="w") do xf
        sheet1 = XLSX.addsheet!(xf, "loads")
        XLSX.writetable!(sheet1, Tables.columntable(data))
    end
end

"""
Merge a `compute_ac_pf` solution ("bus" vm/va, "gen" pg/qg) into a copy of
`test_case` and write the result to `<out_dir>/<datapoint>.json`. Because the
output is a full PowerModels case dict (perturbed parameters + solved
setpoints baked in), it can be read straight back in later with
`PowerModels.parse_file`/`PowerModels.parse_json`.
"""
function store_datapoint!(test_case, solution, datapoint, out_dir)
    out_case = deepcopy(test_case)
    for (ind, val) in solution["gen"]
        out_case["gen"][ind]["pg"] = val["pg"]
        out_case["gen"][ind]["qg"] = val["qg"]
    end
    for (ind, val) in solution["bus"]
        out_case["bus"][ind]["va"] = val["va"]
        out_case["bus"][ind]["vm"] = val["vm"]
    end
    out_case["datapoint"] = datapoint
    filename = joinpath(out_dir, "$datapoint.json")
    # pretty-print (4-space indent) so the file is easy to read by eye; keeps
    # the same Inf/NaN -> "null" handling PowerModels.export_file uses so the
    # file still round-trips through PowerModels.parse_file
    JSON.json(filename, out_case; pretty = 4, allownan = true, inf = "null", ninf = "null", nan = "null")
    return filename
end

"""
Run one perturb + DC-OPF-feasibility + AC-PF-feasibility attempt, starting
from a fresh copy of `base_case`. Seeded internally from `seed` first thing,
so a given attempt index always draws the same perturbation regardless of
when/how often `generate_data` has been called before it.

Returns `(:success, pert_case, solution)` on a feasible attempt, or
`(:dcopf_infeasible,)` / `(:acpf_infeasible,)` on failure at that stage.
"""
function _generate_one_attempt(base_case, max_pd, pert_config, tap_changing_ids, phase_shifting_ids, ipopt, grainger, seed, log_level)
    PowerModels.logger_config!(log_level)
    Random.seed!(seed)
    # start every attempt from a fresh copy of the unperturbed base case
    pert_case = deepcopy(base_case)

    # apply perturbations in place, using the values from pert_config
    loads = zeros(length(pert_case["load"]))
    qd_loads = zeros(length(pert_case["load"]))
    perturb_load!(pert_case, loads, pert_config["load"]["global_scale_delta"], max_pd;
                  qd_loads = qd_loads, jitter_delta = pert_config["load"]["jitter_delta"])
    perturb_gen!(pert_case; gen_fraction = pert_config["gen"]["squeeze_fraction"],
                 max_squeeze_pct = pert_config["gen"]["max_squeeze_pct"])
    shuffle_gencost!(pert_case; shuffle_fraction = pert_config["gencost"]["shuffle_fraction"])
    vsqueeze!(pert_case; bus_fraction = pert_config["vsqueeze"]["bus_fraction"],
             max_squeeze_pct = pert_config["vsqueeze"]["max_squeeze_pct"])
    vasqueeze!(pert_case; line_fraction = pert_config["vasqueeze"]["line_fraction"],
              max_squeeze_pct = pert_config["vasqueeze"]["max_squeeze_pct"])
    thermal_squeeze!(pert_case; line_fraction = pert_config["thermal_squeeze"]["line_fraction"],
                     max_squeeze_pct = pert_config["thermal_squeeze"]["max_squeeze_pct"])
    perturb_sus!(pert_case; bs_delta = pert_config["shunt"]["bs_delta"],
                shunt_fraction = pert_config["shunt"]["shunt_fraction"])
    perturb_tap!(pert_case, tap_changing_ids; max_steps = pert_config["tap"]["max_steps"],
                tap_fraction = pert_config["tap"]["tap_fraction"])
    perturb_shift!(pert_case, phase_shifting_ids; max_steps = pert_config["shift"]["max_steps"],
                   shift_fraction = pert_config["shift"]["shift_fraction"])
    pert_genstatus!(pert_case; num_gens = pert_config["genstatus"]["num_gens"])

    # verify test case is dcopf-feasible
    model_dc = PowerModels.solve_dc_opf(pert_case, ipopt)
    if model_dc["termination_status"] != LOCALLY_SOLVED
        return (:dcopf_infeasible,)
    end
    # place dc gen setpoints into model
    for (ind, val) in model_dc["solution"]["gen"]
        pert_case["gen"][ind]["pg"] = val["pg"]
    end
    pert_linestatus!(pert_case; num_lines = pert_config["linestatus"]["num_lines"])
    # solve traditional ac-pf
    res = PowerModels.compute_ac_pf(pert_case, grainger=grainger, mapping=true)
    if !res["termination_status"]
        return (:acpf_infeasible,)
    end
    return (:success, pert_case, res["solution"])
end

"""
    generate_data(test_case, num_points, case_name, out_name; kwargs...)

Generates up to `2*num_points` perturbation attempts, one at a time (via
`_generate_one_attempt`), stopping once `num_points` feasible datapoints have
been written or `2*num_points` attempts have been made, whichever comes
first.
"""
function generate_data(test_case, num_points, case_name, out_name; pert_config_path::Union{Nothing, AbstractString} = "default_pert.json", log_level::String = "warn", grainger::Bool = true)
    PowerModels.logger_config!(log_level)
    # Ipopt's console output (its banner + per-solve summary) is a completely
    # separate channel from PowerModels' logger above -- it's the solver's
    # own C library printing directly to stdout via the "print_level" MOI
    # option, not something `PowerModels.logger_config!` has any control
    # over. Tie it to the same log_level so "quiet" actually means quiet.
    ipopt_print_level = log_level == "error" ? 0 : 1
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => ipopt_print_level)
    # pert_config_path is just a filename (or an absolute path) -- resolve it
    # against the stored perturbation_configs directory so this works
    # regardless of the shell's current working directory
    resolved_pert_config_path = pert_config_path === nothing ? nothing : joinpath(PERTURBATION_CONFIGS_DIR, pert_config_path)
    pert_config = load_perturbation_config(resolved_pert_config_path)

    # og_pd/og_qd and max_pd are computed once off of the untouched base case;
    # every attempt perturbs a fresh deepcopy of this base case, so
    # perturbations never accumulate/carry over between attempts
    base_case, max_pd = prepare_test_case_perturbations(deepcopy(test_case))
    # classified once off the base case; every attempt perturbs a fresh
    # deepcopy of base_case with the same branch IDs, so these stay valid
    tap_changing_ids, phase_shifting_ids = classify_transformers(base_case)

    out_dir = joinpath(TESTCASE_PATH, "data", case_name, out_name)
    mkpath(out_dir)

    # resume numbering after whatever's already in out_dir instead of
    # starting at 0 and overwriting existing datapoint files. Only files
    # named as a bare integer (the store_datapoint! naming convention) count
    # -- this is the same filter solution_analysis.jl's analyze_dataset uses
    # to find datapoint files, so a leftover violation_summary.json (or
    # anything else) doesn't get mistaken for one.
    existing_samples = filter(f -> occursin(r"^\d+\.json$", f), readdir(out_dir))
    curr_samps = isempty(existing_samples) ? 0 : maximum(parse(Int, splitext(f)[1]) for f in existing_samples) + 1

    counter = 0
    infeas_dcopf = 0
    infeas_acpf = 0
    total_counter = 0
    max_attempts_total = 2 * num_points
    while counter < num_points && total_counter < max_attempts_total
        total_counter += 1
        result = _generate_one_attempt(base_case, max_pd, pert_config,
                        tap_changing_ids, phase_shifting_ids, ipopt, grainger, total_counter, log_level)
        if result[1] == :dcopf_infeasible
            infeas_dcopf += 1
        elseif result[1] == :acpf_infeasible
            infeas_acpf += 1
        else
            _, pert_case, solution = result
            store_datapoint!(pert_case, solution, curr_samps + counter, out_dir)
            counter += 1
        end
        if total_counter % 10 == 0
            println("counter = $counter, dcopf_counter = $infeas_dcopf, acpf_counter = $infeas_acpf, total = $total_counter")
        end
    end
    println("generate_data: wrote $counter datapoints (indices $curr_samps:$(curr_samps + counter - 1)) to $out_dir ($infeas_dcopf dcopf-infeasible, $infeas_acpf acpf-infeasible skipped)")
    return counter
end

function main()
    PowerModels.logger_config!("error")
    PERT_NAME = "extreme_pert"
    for CASE_NAME in [ "case7336"]
        file_pth = joinpath(DATA_PATH, "test_cases/data/$CASE_NAME/$PERT_NAME")
        println("running opf for case $CASE_NAME (no device)...")
        generate_optimal_dataset(file_pth; device=false)
        println("running opf for case $CASE_NAME (device)...")
        generate_optimal_dataset(file_pth; device=true)
    end

    for CASE_NAME in ["case300","case2869_pegase" ]
        println("generating data for $CASE_NAME...")
        # CASE_NAME = "case7336"
        PERT_NAME = "extreme_pert"
        file_pth = joinpath(DATA_PATH, "test_cases/network_info/$CASE_NAME/$(CASE_NAME).m")
        test_case = PowerModels.parse_file(file_pth)
        test_case = prepare_test_case(test_case, CASE_NAME, file_pth)
        counter = generate_data(test_case, 1000, CASE_NAME, PERT_NAME;
                            pert_config_path="$PERT_NAME.json",
                            log_level = "error"
                            )
    end



    # # calculate delta
    # max_pg = sum([gen["pmax"] for gen in values(test_case["gen"])])
    # base_load = sum([load["pd"] for load in values(test_case["load"])])
    # delta = round(0.85*max_pg/base_load - 1, digits=2)
    # delta -= 0.03

    # # pull in loads and generate dataset
    # run_dict = Dict("pf_types" => ["mbuses", "qlim", "baseline"],
    #                 "obo" => [0,1], "grainger" => [0,1],
    #                 "swap_techniques" => ["nearest_gen", "qv_inv", "sensitivity_score"],
    #                 # sensitivity_score-specific knobs (0=off, 1=on); other techniques ignore.
    #                 "use_smw_warmstart"      => [0, 1],
    #                 "score_collateral_aware" => [0],
    #             )
    # load_data = DataFrame(XLSX.readtable(joinpath(TESTCASE_PATH, "data/$(CASE_NAME)/loads/$delta.xlsx"), "loads"))
    # generate_solutions(CASE_NAME, delta, test_case, load_data, file_pth, run_dict; num_samples = 10, write_out = true)
end
