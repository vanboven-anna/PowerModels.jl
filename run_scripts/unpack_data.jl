#!/usr/bin/env julia
# read a dataset.h5 written by generate_dataset.jl's H5Writer back into PowerModels case dicts

using JSON
using HDF5
using Printf

"""
Case dict for sample `row`: static fields from the group attrs, varying
fields sliced out of their [n_samples, n_components] matrices.
"""
function _build_sample(fid, row::Int)
    # dicttype is load-bearing: PowerModels dispatches on Dict{String}
    case = JSON.parse(attrs(fid)["toplevel_json"]; dicttype = Dict{String, Any})
    components = attrs(fid)["components"]
    case["datapoint"] = Int(read(fid["datapoint"])[row])

    for comp in components
        if !haskey(fid, comp)
            case[comp] = Dict{String, Any}()   # empty in the source (dcline/storage/...)
            continue
        end
        g = fid[comp]
        ids = attrs(g)["ids"]
        static = JSON.parse(attrs(g)["static_json"]; dicttype = Dict{String, Any})

        comp_dict = Dict{String, Any}()
        for id in ids
            entry = Dict{String, Any}()
            for (f, byid) in static
                haskey(byid, id) && (entry[f] = byid[id])
            end
            comp_dict[id] = entry
        end

        vg = g["varying"]
        for f in keys(vg)
            endswith(f, "__present") && continue      # presence mask, read below
            d = vg[f]
            kind = attrs(d)["kind"]
            # a field's own ids; not every component carries every field
            fids = attrs(d)["ids"]
            # presence varies per sample too, so restore exactly what was there
            present = attrs(d)["masked"] == 1 ? vg["$(f)__present"][row, :] .!= 0 :
                                                trues(length(fids))
            if kind == "json"
                vals = d[row, :]
                for (j, id) in enumerate(fids)
                    present[j] && (comp_dict[id][f] = JSON.parse(vals[j]; dicttype = Dict{String, Any}))
                end
            elseif kind == "vec"
                vals = d[row, :, :]
                for (j, id) in enumerate(fids)
                    present[j] && (comp_dict[id][f] = collect(vals[j, :]))
                end
            else
                vals = d[row, :]
                for (j, id) in enumerate(fids)
                    present[j] && (comp_dict[id][f] = vals[j])
                end
            end
        end
        case[comp] = comp_dict
    end
    return case
end

"""
    unpack_datapoint(h5_path, datapoint) -> Dict

Return a single datapoint (by its original `<n>.json` index) as a
PowerModels-style Dict.
"""
function unpack_datapoint(h5_path, datapoint::Integer)
    h5open(h5_path, "r") do fid
        dps = read(fid["datapoint"])
        row = findfirst(==(datapoint), dps)
        row === nothing && error("unpack_data: datapoint $datapoint not in $h5_path")
        return _build_sample(fid, row)
    end
end

# directory accessors: everything downstream reads dataset.h5 or legacy <n>.json through these

const DATASET_H5 = "dataset.h5"

dataset_h5_path(dir) = joinpath(dir, DATASET_H5)
has_dataset_h5(dir) = isfile(dataset_h5_path(dir))

"""
    dataset_datapoints(dir) -> Vector{Int}

Sorted datapoint indices in `dir`, from `dataset.h5` or legacy `<n>.json`.
Empty if it holds neither.
"""
function dataset_datapoints(dir)
    isdir(dir) || return Int[]
    if has_dataset_h5(dir)
        return sort(Int.(h5open(f -> read(f["datapoint"]), dataset_h5_path(dir), "r")))
    end
    return sort([parse(Int, splitext(f)[1])
                 for f in readdir(dir) if occursin(r"^\d+\.json$", f)])
end

"""
    load_datapoint(dir, dp) -> Dict{String,Any}

One datapoint from `dir` as a case dict, from `dataset.h5` or `<dp>.json`.
"""
function load_datapoint(dir, dp::Integer)
    if has_dataset_h5(dir)
        return unpack_datapoint(dataset_h5_path(dir), dp)
    end
    p = joinpath(dir, "$dp.json")
    isfile(p) || error("load_datapoint: neither $(DATASET_H5) nor $dp.json in $dir")
    return JSON.parse(read(p, String); dicttype = Dict{String, Any})
end

"""
    read_timing(h5_path) -> Dict

Per-datapoint solve times in seconds: `datapoint`, `acpf_time`, `dcopf_time`.
Empty for archives that recorded none.
"""
function read_timing(h5_path)
    h5open(h5_path, "r") do fid
        out = Dict{String, Vector}()
        for f in ("datapoint", "acpf_time", "dcopf_time")
            haskey(fid, f) && (out[f] = read(fid[f]))
        end
        return out
    end
end

"""
    unpack_dataset(h5_path, out_dir; datapoints = nothing) -> Vector{String}

Write datapoints back out as `<n>.json`; `nothing` writes all of them.
CLI: `julia --project=. run_scripts/unpack_data.jl <in.h5> <out_dir> [dp ...]`
"""
function unpack_dataset(h5_path, out_dir; datapoints = nothing)
    mkpath(out_dir)
    written = String[]
    h5open(h5_path, "r") do fid
        dps = read(fid["datapoint"])
        rows = datapoints === nothing ? (1:length(dps)) :
               [something(findfirst(==(dp), dps),
                          error("unpack_data: datapoint $dp not in $h5_path")) for dp in datapoints]
        for row in rows
            case = _build_sample(fid, row)
            path = joinpath(out_dir, "$(case["datapoint"]).json")
            # store_datapoint!'s settings, so these round-trip through parse_file
            JSON.json(path, case; pretty = 4, allownan = true,
                      inf = "null", ninf = "null", nan = "null")
            push!(written, path)
        end
    end
    @printf("unpack_data: wrote %d datapoints to %s\n", length(written), out_dir)
    return written
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) >= 2 || error("usage: julia unpack_data.jl <in.h5> <out_dir> [datapoint ...]")
    h5_path, out_dir = ARGS[1], ARGS[2]
    dps = length(ARGS) > 2 ? parse.(Int, ARGS[3:end]) : nothing
    unpack_dataset(h5_path, out_dir; datapoints = dps)
end
