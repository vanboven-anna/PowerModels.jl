using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
# Pkg.instantiate
using Revise
using PowerModels
include("../config.jl")
include("./find_nearest_gens.jl")
# load_datapoint / dataset_datapoints, and the dataset.h5 layout H5Writer below writes
include("./unpack_data.jl")
push!(LOAD_PATH, DATA_PATH)
using Infiltrator
Infiltrator.toggle_async_check(false)
using DataFrames
using OrderedCollections
using DataStructures
using JSON
using HDF5
using XLSX
using Ipopt
using Infiltrator
using JuMP
# using Plots
using LinearAlgebra
using Distributions
using Random
using Graphs
# NOTE: the RNG is deliberately unseeded -- a seed would make resumed batches repeat samples

# H5Writer / append_batch!: streaming writer for the dataset.h5 unpack_data.jl reads back

const H5_CHUNK_ROWS = 64     # samples per chunk; matches the archives already written
const H5_DEFLATE = 4
const H5_COL_CHUNK = 256     # samples per chunk for the 1-D per-sample columns

mutable struct H5Writer
    path::String
    case_name::String
    source_dir::String
    n::Int      # samples in the archive, including any an earlier run left there
end

"""
    H5Writer(path; case_name = "", source_dir = "")

Writer for the `dataset.h5` at `path`, appending to it if it exists so a
stopped run resumes. `case_name` defaults to the first sample's `name`.
"""
function H5Writer(path::AbstractString; case_name = "", source_dir = "")
    mkpath(dirname(path))
    n = isfile(path) ? length(h5open(f -> read(f["datapoint"]), path, "r")) : 0
    return H5Writer(String(path), String(case_name),
                    isempty(source_dir) ? dirname(abspath(path)) : String(source_dir), n)
end

# a case key holding component dicts rather than a scalar; empty dcline/storage count
_h5_is_component(v) = v isa AbstractDict && all(x -> x isa AbstractDict, values(v))

function _h5_split_case(case)
    comps, top = String[], Dict{String, Any}()
    for (k, v) in case
        k == "datapoint" && continue           # kept in its own root dataset
        _h5_is_component(v) ? push!(comps, k) : (top[k] = v)
    end
    return sort(comps), top
end

# numeric order where the ids are numbers ("2" before "10"), as PowerModels ids are
_h5_sorted_ids(ids) = sort(collect(ids);
                           by = id -> (something(tryparse(Int, id), typemax(Int)), id))

_h5_json(v) = JSON.json(v; allownan = true, inf = "null", ninf = "null", nan = "null")

"""
Storage kind for a field's values: `int`, `float`, `vec` (equal-length
numeric vectors), or `json` for anything else.
"""
function _h5_kind(vals)
    isempty(vals) && return "json"
    all(v -> v isa Integer && !(v isa Bool), vals) && return "int"
    all(v -> v isa Real && !(v isa Bool), vals) && return "float"
    if all(v -> v isa AbstractVector && all(x -> x isa Real && !(x isa Bool), v), vals) &&
       allequal(length(v) for v in vals)
        return "vec"
    end
    return "json"
end

_h5_veclen(kind, vals) = kind == "vec" ? length(first(vals)) : 0

function _h5_encode(kind, v)
    if kind == "int"
        v isa Integer && return Int64(v)
        (v isa Real && isinteger(v)) && return Int64(v)
        error("append_batch!: $v does not fit the Int64 column it belongs to")
    elseif kind == "float"
        return Float64(v)
    elseif kind == "vec"
        return Float64.(collect(v))
    end
    return _h5_json(v)
end

_h5_fill(kind) = kind == "int" ? Int64(0) : kind == "json" ? "null" : NaN

# every sample's `field` values over `fids`, plus the mask of which samples had it
function _h5_gather(cases, comp, field, fids)
    m, nid = length(cases), length(fids)
    vals = Vector{Any}(undef, m * nid)
    pres = falses(m, nid)
    for (i, case) in enumerate(cases)
        cd = case[comp]
        for (j, id) in enumerate(fids)
            e = get(cd, id, nothing)
            if e !== nothing && haskey(e, field)
                vals[(j - 1) * m + i] = e[field]
                pres[i, j] = true
            end
        end
    end
    return reshape(vals, m, nid), pres
end

_h5_present_vals(vals, pres) = [vals[i] for i in eachindex(pres) if pres[i]]

function _h5_matrix(kind, vals, pres, veclen)
    m, nid = size(vals)
    if kind == "vec"
        out = fill(NaN, m, nid, veclen)
        for i in 1:m, j in 1:nid
            pres[i, j] && (out[i, j, :] = _h5_encode(kind, vals[i, j]))
        end
        return out
    end
    T = kind == "int" ? Int64 : kind == "json" ? String : Float64
    out = fill(_h5_fill(kind), m, nid)
    out = convert(Matrix{T}, out)
    for i in 1:m, j in 1:nid
        pres[i, j] && (out[i, j] = _h5_encode(kind, vals[i, j]))
    end
    return out
end

function _h5_extend_write!(d, from::Int, data)
    rows = (from + 1):(from + size(data, 1))
    if ndims(data) == 3
        HDF5.set_extent_dims(d, (last(rows), size(data, 2), size(data, 3)))
        d[rows, :, :] = data
    elseif ndims(data) == 2
        HDF5.set_extent_dims(d, (last(rows), size(data, 2)))
        d[rows, :] = data
    else
        HDF5.set_extent_dims(d, (last(rows),))
        d[rows] = data
    end
    return d
end

function _h5_append_col!(fid, name, v, n0)
    if !haskey(fid, name)
        d = create_dataset(fid, name, datatype(eltype(v)),
                           dataspace((0,); max_dims = (-1,));
                           chunk = (H5_COL_CHUNK,), deflate = H5_DEFLATE)
        # pad an archive whose earlier run didn't record this column
        n0 > 0 && _h5_extend_write!(d, 0, fill(zero(eltype(v)), n0))
    end
    _h5_extend_write!(fid[name], n0, v)
end

function _h5_create_varying!(vg, field, kind, fids, veclen, n_backfill)
    nid = length(fids)
    T = kind == "int" ? Int64 : kind == "json" ? String : Float64
    dims = kind == "vec" ? (0, nid, veclen) : (0, nid)
    maxd = kind == "vec" ? (-1, nid, veclen) : (-1, nid)
    chunk = kind == "vec" ? (H5_CHUNK_ROWS, nid, veclen) : (H5_CHUNK_ROWS, nid)
    d = create_dataset(vg, field, datatype(T), dataspace(dims; max_dims = maxd);
                       chunk = chunk, deflate = H5_DEFLATE)
    attrs(d)["kind"] = kind
    attrs(d)["ids"] = fids
    attrs(d)["masked"] = 1
    p = create_dataset(vg, "$(field)__present", datatype(UInt8),
                       dataspace((0, nid); max_dims = (-1, nid));
                       chunk = (H5_CHUNK_ROWS, nid), deflate = H5_DEFLATE)
    # a field seen only partway through the run: earlier samples are masked absent
    if n_backfill > 0
        _h5_write_rows!(d, p, 0, falses(n_backfill, nid), kind, veclen,
                        Matrix{Any}(undef, n_backfill, nid))
    end
    return d, p
end

function _h5_write_rows!(d, p, from, pres, kind, veclen, vals)
    _h5_extend_write!(d, from, _h5_matrix(kind, vals, pres, veclen))
    _h5_extend_write!(p, from, UInt8.(pres))
end

# a static field this batch disagreed with: move it to varying, backfilling its old value
function _h5_promote!(vg, field, static_byid, fids, n0)
    vals0 = [static_byid[id] for id in fids if haskey(static_byid, id)]
    kind = _h5_kind(vals0)
    veclen = _h5_veclen(kind, vals0)
    d, p = _h5_create_varying!(vg, field, kind, fids, veclen, 0)
    if n0 > 0
        back = Matrix{Any}(undef, n0, length(fids))
        pres = falses(n0, length(fids))
        for (j, id) in enumerate(fids)
            haskey(static_byid, id) || continue
            for i in 1:n0
                back[i, j] = static_byid[id]
                pres[i, j] = true
            end
        end
        _h5_write_rows!(d, p, 0, pres, kind, veclen, back)
    end
    return d, p
end

_h5_set_attr!(obj, k, v) = (haskey(attrs(obj), k) && delete_attribute(obj, k); attrs(obj)[k] = v)

"""
One component group for this batch. The first batch decides what is static;
after that the schema on disk wins unless it can no longer describe a field.
"""
function _h5_append_component!(fid, comp, cases, n0)
    m = length(cases)
    ids = _h5_sorted_ids(union((keys(case[comp]) for case in cases)...))
    fresh = !haskey(fid, comp)
    if fresh
        isempty(ids) && return          # empty in every sample: no group, reader gives Dict()
        g = create_group(fid, comp)
        attrs(g)["ids"] = ids
        attrs(g)["static_json"] = "{}"
        create_group(g, "varying")
    end
    g = fid[comp]
    stored_ids = attrs(g)["ids"]
    extra = setdiff(ids, stored_ids)
    isempty(extra) ||
        error("append_batch!: $comp gained id(s) $(sort(collect(extra))) part-way through " *
              "$(HDF5.filename(fid)) -- the archive is built around a fixed set of components")
    static = JSON.parse(attrs(g)["static_json"]; dicttype = Dict{String, Any})
    vg = g["varying"]
    existing = [f for f in keys(vg) if !endswith(f, "__present")]

    batch_fields = Set{String}()
    for case in cases, e in values(case[comp])
        union!(batch_fields, keys(e))
    end
    # every varying dataset grows by `m` rows, or the sample axis stops lining up
    static_changed = false

    for field in sort(collect(union(batch_fields, Set(existing), Set(keys(static)))))
        if haskey(static, field) && !(field in existing)
            byid = static[field]
            fids = _h5_sorted_ids(keys(byid))
            vals, pres = _h5_gather(cases, comp, field, stored_ids)
            holds = all(1:m) do i
                all(enumerate(stored_ids)) do (j, id)
                    pres[i, j] ? (haskey(byid, id) && isequal(vals[i, j], byid[id])) :
                                 !haskey(byid, id)
                end
            end
            holds && continue                       # still static, nothing to write
            _h5_promote!(vg, field, byid, fids, n0)
            delete!(static, field)
            static_changed = true
            push!(existing, field)
        end

        if field in existing
            d = vg[field]
            fids = attrs(d)["ids"]
            kind = attrs(d)["kind"]
            veclen = kind == "vec" ? size(d, 3) : 0
            vals, pres = _h5_gather(cases, comp, field, fids)
            _h5_write_rows!(d, vg["$(field)__present"], n0, pres, kind, veclen, vals)
            continue
        end

        # first time this field is seen at all
        fids = _h5_sorted_ids([id for id in stored_ids
                               if any(haskey(get(case[comp], id, Dict()), field) for case in cases)])
        vals, pres = _h5_gather(cases, comp, field, fids)
        seen = _h5_present_vals(vals, pres)
        kind = _h5_kind(seen)
        veclen = _h5_veclen(kind, seen)
        # on the first batch, a field identical in every sample goes to static_json
        if fresh && all(pres) && all(i -> isequal(vals[i, :], vals[1, :]), 1:m)
            static[field] = Dict{String, Any}(id => vals[1, j] for (j, id) in enumerate(fids))
            static_changed = true
            continue
        end
        d, p = _h5_create_varying!(vg, field, kind, fids, veclen, n0)
        _h5_write_rows!(d, p, n0, pres, kind, veclen, vals)
    end

    static_changed && _h5_set_attr!(g, "static_json", _h5_json(static))
    return nothing
end

function _h5_check_toplevel(fid, top)
    stored = JSON.parse(attrs(fid)["toplevel_json"]; dicttype = Dict{String, Any})
    bad = [k for k in union(keys(stored), keys(top)) if !isequal(get(stored, k, nothing), get(top, k, nothing))]
    isempty(bad) ||
        error("append_batch!: case-wide field(s) $(sort(bad)) differ from the ones " *
              "$(HDF5.filename(fid)) was started with -- these are stored once for the " *
              "whole archive, so samples with a different network belong in their own file")
end

"""
    append_batch!(writer, cases, acpf_time, dcopf_time) -> Int

Append case dicts (each carrying its `"datapoint"` index) and their solve
times, returning the new total sample count.
"""
function append_batch!(w::H5Writer, cases::AbstractVector, acpf_time, dcopf_time)
    isempty(cases) && return w.n
    (length(acpf_time) == length(cases) && length(dcopf_time) == length(cases)) ||
        error("append_batch!: got $(length(cases)) cases but $(length(acpf_time)) acpf / " *
              "$(length(dcopf_time)) dcopf times")
    comps, top = _h5_split_case(cases[1])

    h5open(w.path, isfile(w.path) ? "r+" : "w") do fid
        n0 = haskey(fid, "datapoint") ? length(fid["datapoint"]) : 0
        if !haskey(attrs(fid), "toplevel_json")
            attrs(fid)["format_version"] = 1
            attrs(fid)["streaming"] = 1
            attrs(fid)["case_name"] = isempty(w.case_name) ?
                                      string(get(cases[1], "name", "")) : w.case_name
            attrs(fid)["source_dir"] = w.source_dir
            attrs(fid)["components"] = comps
            attrs(fid)["toplevel_json"] = _h5_json(top)
        end
        _h5_check_toplevel(fid, top)
        known = attrs(fid)["components"]
        union(known, comps) == known || _h5_set_attr!(fid, "components", sort(union(known, comps)))

        _h5_append_col!(fid, "datapoint", Int64[Int(c["datapoint"]) for c in cases], n0)
        _h5_append_col!(fid, "acpf_time", Float64.(collect(acpf_time)), n0)
        _h5_append_col!(fid, "dcopf_time", Float64.(collect(dcopf_time)), n0)
        for comp in comps
            _h5_append_component!(fid, comp, cases, n0)
        end
        w.n = n0 + length(cases)
    end
    return w.n
end

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
Draw a perturbed load satisfying `_verify_loads_`, up to `max_attempts`
times, shrinking `delta`/`jitter_delta` on each retry. Iterative, not
recursive: an always-infeasible case must error, not overflow the stack.
A case infeasible at zero perturbation still fails here.
"""
function perturb_load!(test_case, loads, delta, max_pd; qd_loads = nothing, jitter_delta = 0.05, max_attempts = 20)
    # get max and min total generation
    max_gen = sum([gen["pmax"] for gen in values(test_case["gen"])])
    min_gen = sum([gen["pmin"] for gen in values(test_case["gen"])])
    max_qg = qd_loads === nothing ? nothing : sum([gen["qmax"] for gen in values(test_case["gen"])])
    min_qg = qd_loads === nothing ? nothing : sum([gen["qmin"] for gen in values(test_case["gen"])])

    for attempt in 1:max_attempts
        # shrink the perturbation range each retry, down to 1/max_attempts on the last
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

# perturbation config: fractions come from perturbation_configs/, overridable per key

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

# perturbation functions: each mutates `test_case` in place and returns it, so they chain

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
Squeeze [pmin, pmax] on a `gen_fraction` subset of generators, each by a
percentage drawn from [0, max_squeeze_pct] and applied symmetrically.
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
Squeeze [vmin, vmax] on a `bus_fraction` subset of buses, each by a
percentage drawn from [0, max_squeeze_pct] and applied symmetrically.
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
Squeeze [angmin, angmax] on a `line_fraction` subset of branches, each by a
percentage drawn from [0, max_squeeze_pct] and applied symmetrically.
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
Squeeze `rate_a` on a `line_fraction` subset of branches, each scaled by
(1 - squeeze_pct) for a percentage drawn from [0, max_squeeze_pct].
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
Outages are skipped on networks this size or smaller: a small case has too
little redundancy for a single outage to leave a solvable network.
"""
const OUTAGE_MIN_BUSES = 300

"""
Turn off `num_gens` in-service generators (a count, not a fraction), never
the slack. Call before DC-OPF. No-op at `OUTAGE_MIN_BUSES` buses or fewer.
`correct_bus_types!` afterwards demotes any PV bus left with no generator,
which the AC-PF solver would otherwise hit a `KeyError` on.
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
Turn off `num_lines` in-service branches (a count, not a fraction), modeling
an outage after dispatch. Candidates are accepted one at a time and only if
the endpoints stay connected, so no combination of removals islands a bus
(the AC-PF solver crashes on one). No-op at `OUTAGE_MIN_BUSES` or fewer.
"""
function pert_linestatus!(test_case; num_lines::Integer = 0)
    num_lines <= 0 && return test_case
    length(test_case["bus"]) <= OUTAGE_MIN_BUSES && return test_case
    in_service = [(ind, branch) for (ind, branch) in test_case["branch"] if get(branch, "br_status", 1) != 0]
    isempty(in_service) && return test_case

    # SimpleGraph collapses parallel branches, so pair_remaining tracks the real count
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

# real tap changers move in discrete steps, so draw from a step grid, not a Uniform
const TAP_STEP_PCT = 0.006   # 0.6% per discrete tap-changer step
const SHIFT_STEP_DEG = 2.0   # degrees per discrete phase-shifter step
const SHIFT_STEP_RAD = deg2rad(SHIFT_STEP_DEG)

"""
Transformer branch IDs split into tap-changing (`tap != 1.0`) and
phase-shifting (`shift` not 0 or a full rotation); the two may overlap.
Call once on the base case so every sample perturbs the same sets.
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
Scale the tap of a `tap_fraction` subset of `eligible_ids` by an integer
multiple in `-max_steps:max_steps` of `step_pct`, as a mechanical tap changer
moves (default 0.6%, a ±10% range over 33 positions).
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
Shift the phase of a `shift_fraction` subset of `eligible_ids` by an integer
multiple in `-max_steps:max_steps` of `step_rad` (additive, default 2
degrees). `shift` is stored in radians.
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
Tuning bounds and discrete setpoint grids on `tap_ids`/`shift_ids`, plus
shunt susceptance bounds, for `build_dc_ac_device_pf`. Every flagged branch
shares one grid centered on nominal, as real tap positions are. Shunt
bmin/bmax stay relative to each `bs`.
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

Write a solution back into `test_case`: bus vm/va, gen pg/qg (vg to the solved
vm at its bus), solved tap/shift snapped to its grid. Iterates the solution,
not `test_case`, since inactive components are absent from it.
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

AC-feasible point closest to the generator voltage setpoints, applied back into
`test_case`. `device = true` first solves with tap/shift/shunt free, snaps them
to their grid, then resolves. Returns `(test_case, result)` for the last solve.
"""
function solve_dc_ac_pf!(test_case, optimizer; device::Bool = false)
    if device
        tap_ids, shift_ids = classify_transformers(test_case)
        prepare_transformer_adjustments(test_case, tap_ids, shift_ids)
        device_result = PowerModels.solve_dc_ac_device_pf(test_case, optimizer)
        apply_solution!(test_case, device_result)
        # the build leaks a non-serializable penalty dict into test_case (pm.data is it)
        delete!(test_case, "soft_bound_penalty_dict")
    end
    result = PowerModels.solve_dc_ac_pf(test_case, optimizer)
    apply_solution!(test_case, result)
    delete!(test_case, "soft_bound_penalty_dict")
    return test_case, result
end

"""
    generate_optimal_dataset(dataset_path; device = false, smoke = false)

`solve_dc_ac_pf!` over `<dataset_path>/baseline_acpf`, into the sibling
`dc_ac_device_pf/` or `dc_ac_pf/` under the same index; unconverged datapoints
are skipped and printed. Also writes `metadata.xlsx`, one row per attempt.
"""
function generate_optimal_dataset(dataset_path; device::Bool = false, smoke::Bool = false, smoke_n::Integer = 10, log_level::String = "error", batch_size::Int = 100)
    PowerModels.logger_config!(log_level)
    # Ipopt's console output is its own channel, not PowerModels' logger
    ipopt_print_level = log_level == "error" ? 0 : 1
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => ipopt_print_level)

    in_dir = joinpath(dataset_path, "baseline_acpf")
    out_dir = joinpath(dataset_path, device ? "dc_ac_device_pf" : "dc_ac_pf")
    mkpath(out_dir)

    datapoints = dataset_datapoints(in_dir)
    isempty(datapoints) && error("generate_optimal_dataset: no datapoints (dataset.h5 or <n>.json) in $in_dir")

    # smoke-test mode: only the first `smoke_n` datapoints
    if smoke
        datapoints = datapoints[1:min(smoke_n, end)]
        println("generate_optimal_dataset: SMOKE mode -- first $(length(datapoints)) datapoint(s) only: $datapoints")
    end

    metadata = DataFrame(datapoint = Int[], feasible = Bool[], time = Float64[], objective = Float64[])

    # solved cases accumulate into one dataset.h5, flushed every `batch_size`
    writer = H5Writer(joinpath(out_dir, "dataset.h5"))
    batch, batch_time, batch_dc = Any[], Float64[], Float64[]
    flush_batch! = () -> begin
        isempty(batch) && return
        append_batch!(writer, batch, batch_time, batch_dc)
        empty!(batch); empty!(batch_time); empty!(batch_dc); GC.gc()
    end

    written, skipped = 0, Int[]
    for (processed, datapoint) in enumerate(datapoints)
        test_case = load_datapoint(in_dir, datapoint)
        local result
        elapsed = @elapsed (test_case, result) = solve_dc_ac_pf!(test_case, ipopt; device = device)
        solved = result["termination_status"] == LOCALLY_SOLVED
        # qg/vm bounds are soft in these builds, so check the hard limits separately
        feasible = solved && _determine_acpf_feasibility(test_case, Dict())["total_violations"] == 0
        push!(metadata, (datapoint, feasible, elapsed, result["objective"]))

        if !solved
            push!(skipped, datapoint)
        else
            # original index, never renumbered, so it lines up with baseline_acpf
            test_case["datapoint"] = datapoint
            push!(batch, test_case); push!(batch_time, elapsed); push!(batch_dc, 0.0)
            length(batch) >= batch_size && flush_batch!()
            written += 1
        end

        if processed % 50 == 0
            println("generate_optimal_dataset: processed $processed/$(length(datapoints)) (written $written, skipped $(length(skipped))) -- $out_dir")
            flush(stdout)
        end
    end
    flush_batch!()

    metadata_path = joinpath(out_dir, "metadata.xlsx")
    isfile(metadata_path) && rm(metadata_path)
    XLSX.openxlsx(metadata_path, mode = "w") do xf
        sheet = XLSX.addsheet!(xf, "metadata")
        XLSX.writetable!(sheet, Tables.columntable(metadata))
    end

    println("generate_optimal_dataset: wrote $written cases to $(joinpath(out_dir, "dataset.h5"))")
    if !isempty(skipped)
        println("generate_optimal_dataset: skipped $(length(skipped)) unconverged datapoint(s), missing from $out_dir: $(sort(skipped))")
    end
    return out_dir
end

"""
    common_datapoint_files(dir; exclude = String[])

Sorted datapoint indices present in every immediate subdirectory of `dir`,
i.e. those a cross-directory comparison has a match for in each.
"""
function common_datapoint_files(dir; exclude = String[])
    subdirs = filter(d -> isdir(joinpath(dir, d)) && !(d in exclude), readdir(dir))
    isempty(subdirs) && return Int[], subdirs
    index_sets = [Set(dataset_datapoints(joinpath(dir, d))) for d in subdirs]
    # an empty subdirectory would otherwise intersect everything down to nothing
    nonempty = [s for s in index_sets if !isempty(s)]
    isempty(nonempty) && return Int[], subdirs
    return sort(collect(intersect(nonempty...))), subdirs
end

"""
    generate_sensitivity_score_dataset(dataset_path; grainger = true, smoke = false)

`compute_ac_pf_mult_buses(swap_technique = "sensitivity_score")` over every
datapoint common to all subdirectories of `dataset_path`, into `acpf_ss/`;
unconverged datapoints are skipped and printed. Also writes `metadata.xlsx`
with per-datapoint feasibility, time, swap iterations and stop reason.
"""
function generate_sensitivity_score_dataset(dataset_path; grainger::Bool = true, max_acpf::Integer = 50, smoke::Bool = false, smoke_n::Integer = 10, obo = false, log_level::String = "error", batch_size::Int = 100)
    PowerModels.logger_config!(log_level)

    out_name = "acpf_ss"
    out_dir = joinpath(dataset_path, out_name)

    common_files, subdirs = common_datapoint_files(dataset_path; exclude = [out_name])
    if isempty(subdirs)
        error("generate_sensitivity_score_dataset: no subdirectories to scan under $dataset_path")
    end
    if isempty(common_files)
        println("generate_sensitivity_score_dataset: no case file is present in all of $(subdirs) under $dataset_path")
        return out_dir
    end

    mkpath(out_dir)

    # smoke-test mode: only the first `smoke_n` datapoints
    if smoke
        common_files = common_files[1:min(smoke_n, end)]
        println("generate_sensitivity_score_dataset: SMOKE mode -- first $(length(common_files)) datapoint(s) only: $common_files")
    end

    # any sibling will do: they share the perturbed parameters, only setpoints differ
    source_dir = dp -> begin
        "baseline_acpf" in subdirs && return joinpath(dataset_path, "baseline_acpf")
        for d in subdirs
            dp in dataset_datapoints(joinpath(dataset_path, d)) && return joinpath(dataset_path, d)
        end
        error("generate_sensitivity_score_dataset: datapoint $dp vanished from every subdirectory")
    end

    # stop_reason / *_remaining come from compute_ac_pf_mult_buses' instrumentation
    metadata = DataFrame(datapoint = Int[], feasible = Bool[], time = Float64[], swap_iters = Int[],
                         stop_reason = String[], pq_vm_viol_remaining = Int[],
                         pv_qg_viol_remaining = Int[], pv_donors_left = Int[],
                         recipients_no_donor = Int[], recipients_low_sens = Int[])

    writer = H5Writer(joinpath(out_dir, "dataset.h5"))
    ss_batch, ss_time = Any[], Float64[]
    flush_ss! = () -> begin
        isempty(ss_batch) && return
        append_batch!(writer, ss_batch, ss_time, zeros(length(ss_batch)))
        empty!(ss_batch); empty!(ss_time); GC.gc()
    end

    written, skipped = 0, Int[]
    for (processed, datapoint) in enumerate(common_files)
        test_case = load_datapoint(source_dir(datapoint), datapoint)

        # keep_history = false: the history dominates memory on large networks
        elapsed = @elapsed result = PowerModels.compute_ac_pf_mult_buses(test_case;
                        grainger = grainger, swap_technique = "sensitivity_score",
                        max_acpf = max_acpf, enforce_q_lims = true, keep_history = false, obo = obo)

        solution = result["solution"]
        swap_iters = length(get(result, "solution_history", []))
        converged = result["termination_status"] === true &&
                    solution !== nothing && haskey(solution, "bus") &&
                    !any(b -> b["vm"] == -1, values(solution["bus"]))

        stop_reason = get(result, "stop_reason", "unknown")
        fd = get(result, "final_diagnostics", Dict{String,Any}())
        sa = get(fd, "swap_attempt", Dict{String,Any}())
        pq_vm_rem = Int(get(fd, "pq_vm_violations", -1))
        pv_qg_rem = Int(get(fd, "pv_qg_violations", -1))
        pv_donors_left = Int(get(fd, "pv_donor_buses_available", -1))
        recip_no_donor = Int(get(sa, "recipients_no_candidate_donor", -1))
        recip_low_sens = Int(get(sa, "recipients_low_sensitivity", -1))

        # qg/vm bounds stay soft, so convergence alone doesn't mean they held
        feasible = converged &&
                   solution_feasibility(deepcopy(test_case), solution, Dict())["total_violations"] == 0
        push!(metadata, (datapoint, feasible, elapsed, swap_iters, stop_reason,
                         pq_vm_rem, pv_qg_rem, pv_donors_left, recip_no_donor, recip_low_sens))

        if converged
            # the merge store_datapoint! did, batched into dataset.h5
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
            push!(ss_batch, out_case); push!(ss_time, elapsed)
            length(ss_batch) >= batch_size && flush_ss!()
            written += 1
        else
            push!(skipped, datapoint)
        end

        # reclaim the solve's Jacobians now rather than when the GC decides to
        result = nothing
        GC.gc()

        if processed % 50 == 0
            println("generate_sensitivity_score_dataset: processed $processed/$(length(common_files)) (written $written, skipped $(length(skipped))) -- $out_dir")
        end
    end

    flush_ss!()

    metadata_path = joinpath(out_dir, "metadata.xlsx")
    isfile(metadata_path) && rm(metadata_path)
    XLSX.openxlsx(metadata_path, mode = "w") do xf
        sheet = XLSX.addsheet!(xf, "metadata")
        XLSX.writetable!(sheet, Tables.columntable(metadata))
    end

    println("generate_sensitivity_score_dataset: wrote $written cases to $(joinpath(out_dir, "dataset.h5"))")
    if !isempty(skipped)
        println("generate_sensitivity_score_dataset: skipped $(length(skipped)) unconverged datapoint(s), missing from $out_dir: $(sort(skipped))")
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
`rate_a` used in `max_pd` for an unrated branch: far above any real rating,
so it acts as unbounded without being a literal `Inf`.
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
        # a missing rate_a means "unrated", so treat it as unbounded, not as zero
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
                # two extra knobs only sensitivity_score reads; [0] keeps the loop shape
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
    # Ipopt's console output is its own channel, not PowerModels' logger
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
Merge a solution's bus vm/va and gen pg/qg into a copy of `test_case` and
write it to `<out_dir>/<datapoint>.json`, readable back with `parse_file`.
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
    # export_file's Inf/NaN -> null handling, so this round-trips through parse_file
    JSON.json(filename, out_case; pretty = 4, allownan = true, inf = "null", ninf = "null", nan = "null")
    return filename
end

"""
One perturb + DC-OPF + AC-PF attempt off a fresh copy of `base_case`; the
RNG is deliberately not seeded. Returns `(:success, pert_case, solution)`,
`(:dcopf_infeasible,)` or `(:acpf_infeasible,)`.
"""
function _generate_one_attempt(base_case, max_pd, pert_config, tap_changing_ids, phase_shifting_ids, ipopt, grainger, log_level)
    PowerModels.logger_config!(log_level)
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
    dcopf_time = @elapsed model_dc = PowerModels.solve_dc_opf(pert_case, ipopt)
    if model_dc["termination_status"] != LOCALLY_SOLVED
        return (:dcopf_infeasible,)
    end
    # warm start off DC-OPF: pg, angles into va_start, gen vg into vm_start when all have one
    warmstart_pf_from_dc!(pert_case, model_dc["solution"])
    pert_linestatus!(pert_case; num_lines = pert_config["linestatus"]["num_lines"])
    # enforce_q_lims = false: PV->PQ switching often fails to converge on large networks
    acpf_time = @elapsed res = PowerModels.compute_ac_pf(pert_case, grainger = grainger,
                                                         mapping = true, enforce_q_lims = false)
    if !res["termination_status"]
        return (:acpf_infeasible,)
    end
    return (:success, pert_case, res["solution"], acpf_time, dcopf_time)
end

"""
    warmstart_pf_from_dc!(case, dc_solution)

Seed an AC-PF solve from a DC-OPF solution: dispatch into gen `pg`, angles into
`va_start`, gen `vg` into `vm_start` when every in-service generator has one.
"""
function warmstart_pf_from_dc!(case, dc_solution)
    for (i, gen) in get(dc_solution, "gen", Dict())
        haskey(case["gen"], i) && haskey(gen, "pg") && (case["gen"][i]["pg"] = gen["pg"])
    end
    for (i, bus) in get(dc_solution, "bus", Dict())
        haskey(case["bus"], i) && haskey(bus, "va") && (case["bus"][i]["va_start"] = bus["va"])
    end
    live_gens = [g for g in values(case["gen"]) if get(g, "gen_status", 1) != 0]
    if !isempty(live_gens) && all(g -> haskey(g, "vg"), live_gens)
        for g in live_gens
            b = string(g["gen_bus"])
            haskey(case["bus"], b) && (case["bus"][b]["vm_start"] = g["vg"])
        end
    end
    return case
end

"""
    generate_data(test_case, num_points, case_name, out_name; kwargs...)

Perturbation attempts until `baseline_acpf/dataset.h5` holds `num_points`
datapoints or `time_budget_s` runs out. `num_points` is a dataset target, so a
second call tops it up. Written in batches of `batch_size`, with solve times.
"""
function generate_data(test_case, num_points, case_name, out_name;
                       pert_config_path::Union{Nothing, AbstractString} = "default_pert.json",
                       log_level::String = "warn", grainger::Bool = true,
                       batch_size::Int = 100, time_budget_s::Real = Inf)
    PowerModels.logger_config!(log_level)
    # Ipopt prints from its own C library, so tie print_level to log_level too
    ipopt_print_level = log_level == "error" ? 0 : 1
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => ipopt_print_level)
    # resolve the config filename against perturbation_configs, whatever the cwd is
    resolved_pert_config_path = pert_config_path === nothing ? nothing : joinpath(PERTURBATION_CONFIGS_DIR, pert_config_path)
    pert_config = load_perturbation_config(resolved_pert_config_path)

    # off the untouched base case, so perturbations never accumulate between attempts
    base_case, max_pd = prepare_test_case_perturbations(deepcopy(test_case))
    # branch IDs are the same in every attempt, so one classification stays valid
    tap_changing_ids, phase_shifting_ids = classify_transformers(base_case)

    out_dir = joinpath(TESTCASE_PATH, "data", case_name, out_name, "baseline_acpf")
    mkpath(out_dir)
    writer = H5Writer(joinpath(out_dir, "dataset.h5"))

    # resume, not overwrite: num_points is the dataset target, so only the shortfall is run
    existing = dataset_datapoints(out_dir)
    curr_samps = isempty(existing) ? 0 : maximum(existing) + 1
    n_existing = length(existing)
    to_generate = num_points - n_existing
    if to_generate <= 0
        @printf("generate_data: %s already has %d datapoints (target %d) -- nothing to do\n",
                case_name, n_existing, num_points)
        return n_existing
    end
    if n_existing > 0
        @printf("generate_data: %s has %d datapoints, generating %d more to reach %d (new indices start at %d)\n",
                case_name, n_existing, to_generate, num_points, curr_samps)
    end

    counter = 0
    infeas_dcopf = 0
    infeas_acpf = 0
    total_counter = 0
    t0 = time()
    batch, batch_acpf, batch_dcopf = Any[], Float64[], Float64[]

    function flush_batch!()
        isempty(batch) && return
        append_batch!(writer, batch, batch_acpf, batch_dcopf)
        flush(stdout)
        empty!(batch); empty!(batch_acpf); empty!(batch_dcopf)
        GC.gc()
    end

    while counter < to_generate
        total_counter += 1
        result = _generate_one_attempt(base_case, max_pd, pert_config,
                        tap_changing_ids, phase_shifting_ids, ipopt, grainger, log_level)
        if result[1] == :dcopf_infeasible
            infeas_dcopf += 1
        elseif result[1] == :acpf_infeasible
            infeas_acpf += 1
        else
            _, pert_case, solution, acpf_time, dcopf_time = result
            # same merge store_datapoint! did, but into memory rather than a file
            out_case = deepcopy(pert_case)
            for (ind, val) in solution["gen"]
                out_case["gen"][ind]["pg"] = val["pg"]
                out_case["gen"][ind]["qg"] = val["qg"]
            end
            for (ind, val) in solution["bus"]
                out_case["bus"][ind]["va"] = val["va"]
                out_case["bus"][ind]["vm"] = val["vm"]
            end
            out_case["datapoint"] = curr_samps + counter
            push!(batch, out_case); push!(batch_acpf, acpf_time); push!(batch_dcopf, dcopf_time)
            counter += 1
            length(batch) >= batch_size && flush_batch!()
        end
        if total_counter % 10 == 0
            println("counter = $counter, dcopf_counter = $infeas_dcopf, acpf_counter = $infeas_acpf, total = $total_counter")
            flush(stdout)
        end
    
        if time() - t0 > time_budget_s
            @printf("generate_data: time budget (%.0fs) reached for %s after %d datapoints -- stopping\n",
                    time_budget_s, case_name, counter)
            break
        end
    end
    flush_batch!()
    @printf("generate_data: wrote %d datapoints to %s in %.1fs (%d dcopf-infeasible, %d acpf-infeasible skipped)\n",
            writer.n, joinpath(out_dir, "dataset.h5"), time() - t0, infeas_dcopf, infeas_acpf)
    return writer.n
end



function main()
    PowerModels.logger_config!("error")
    PERT_NAME = "extreme_pert"
    # for CASE_NAME in [ "case14", "case57", "case300"]
    #     file_pth = joinpath(DATA_PATH, "test_cases/data/$CASE_NAME/$PERT_NAME")
    #     # println("running opf for case $CASE_NAME (no device)...")
    #     # generate_optimal_dataset(file_pth; device=false, smoke=true)
    #     # println("running opf for case $CASE_NAME (device)...")
    #     # generate_optimal_dataset(file_pth; device=true, smoke=true)
    #     println("running sensitivity_score ac-pf for case $CASE_NAME...")
    #     generate_sensitivity_score_dataset(file_pth; grainger=true, max_acpf=30, smoke=true, obo = true)
    # end
    # for CASE_NAME in [ "case7336"]
    #     file_pth = joinpath(DATA_PATH, "test_cases/data/$CASE_NAME/$PERT_NAME")
    #     println("running opf for case $CASE_NAME (no device)...")
    #     generate_optimal_dataset(file_pth; device=false)
    #     println("running opf for case $CASE_NAME (device)...")
    #     generate_optimal_dataset(file_pth; device=true)
    #     println("running sensitivity_score ac-pf for case $CASE_NAME...")
    #     generate_sensitivity_score_dataset(file_pth; grainger=true)
    # end

    CASES = [ "case9241_pegase"]
    NUM_POINTS = 1000
    BATCH_SIZE = 100
    TIME_BUDGET_S = 1.5 * 60 * 60   # per case; a case that overruns is abandoned

    for CASE_NAME in CASES
        println("\n", "="^70)
        println("generating $NUM_POINTS datapoints for $CASE_NAME ($PERT_NAME)")
        println("="^70)
        flush(stdout)
        try
            file_pth = joinpath(DATA_PATH, "test_cases/network_info/$CASE_NAME/$(CASE_NAME).m")
            test_case = prepare_test_case(PowerModels.parse_file(file_pth), CASE_NAME, file_pth)
            n = generate_data(test_case, NUM_POINTS, CASE_NAME, PERT_NAME;
                              pert_config_path = "$PERT_NAME.json",
                              log_level = "error",
                              batch_size = BATCH_SIZE,
                              time_budget_s = TIME_BUDGET_S)
            println("$CASE_NAME: done -- $n datapoints")
        catch e
            println("$CASE_NAME: FAILED -- $(sprint(showerror, e))")
        end
        GC.gc()
        flush(stdout)
    end

end
