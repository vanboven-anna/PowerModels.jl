# Compares the traditional and grainger AC power-flow implementations across
# every test case under bus_swap_data/test_cases/network_info: parses each
# case's .m file, runs compute_ac_pf with mapping=true and grainger=false vs.
# grainger=true, times each run, plots computation time by test case, and
# writes a report file recording -- per case -- whether the two solutions
# (vm, va, pg, qg for every bus/gen) agree.
#
# Run from the repo root:
#   julia --project=. --startup-file=no test_scripts/test_grainger.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using PowerModels
include(joinpath(@__DIR__, "..", "config.jl"))
using Plots
using Printf
using JSON

PowerModels.silence()

const NETWORK_INFO_DIR = joinpath(DATA_PATH, "test_cases", "network_info")
const SOLUTION_TOL = 1e-6
const PLOT_PATH = joinpath(@__DIR__, "grainger_timing_comparison.png")
const REPORT_PATH = joinpath(@__DIR__, "grainger_comparison_report.json")

# validated colorblind-safe categorical pair (dataviz skill default palette,
# slots 1 and 2)
const COLOR_TRADITIONAL = "#2a78d6"
const COLOR_GRAINGER = "#eb6834"

struct GraingerCaseResult
    case_name::String
    nbus::Int
    time_traditional::Float64
    time_grainger::Float64
    converged_traditional::Bool
    converged_grainger::Bool
    solutions_match::Bool
    max_dvm::Float64
    max_dva::Float64
    max_dpg::Float64
    max_dqg::Float64
    error_msg::String
end

struct GraingerTestCase
    case_name::String
    data::Dict{String,Any}
    nbus::Int
end

"parse every case's .m file under network_info (one per case directory), sorted smallest-to-largest by bus count"
function _load_test_cases()
    cases = GraingerTestCase[]
    for entry in sort(readdir(NETWORK_INFO_DIR))
        dir = joinpath(NETWORK_INFO_DIR, entry)
        isdir(dir) || continue
        m_file = joinpath(dir, "$(entry).m")
        if isfile(m_file)
            data = PowerModels.parse_file(m_file)
            push!(cases, GraingerTestCase(entry, data, length(data["bus"])))
        else
            @warn "skipping $(entry): no $(entry).m file found"
        end
    end
    sort!(cases, by = c -> c.nbus)
    return cases
end

"max absolute difference between two compute_ac_pf solution dicts, across vm, va, pg, qg"
function _solution_diffs(sol_a, sol_b)
    max_dvm = 0.0
    max_dva = 0.0
    for (i, bus_a) in sol_a["bus"]
        bus_b = sol_b["bus"][i]
        max_dvm = max(max_dvm, abs(bus_a["vm"] - bus_b["vm"]))
        max_dva = max(max_dva, abs(bus_a["va"] - bus_b["va"]))
    end
    max_dpg = 0.0
    max_dqg = 0.0
    for (i, gen_a) in sol_a["gen"]
        gen_b = sol_b["gen"][i]
        max_dpg = max(max_dpg, abs(gen_a["pg"] - gen_b["pg"]))
        max_dqg = max(max_dqg, abs(gen_a["qg"] - gen_b["qg"]))
    end
    return max_dvm, max_dva, max_dpg, max_dqg
end

"run both the traditional and grainger compute_ac_pf on one already-parsed case, timing each and diffing the solutions"
function _run_case(case_name, data)
    nbus = length(data["bus"])

    t_trad = @elapsed result_trad = PowerModels.compute_ac_pf(
        deepcopy(data); mapping = true, grainger = false, enforce_q_lims = false)
    t_grainger = @elapsed result_grainger = PowerModels.compute_ac_pf(
        deepcopy(data); mapping = true, grainger = true, enforce_q_lims = false)

    converged_trad = result_trad["termination_status"] == true
    converged_grainger = result_grainger["termination_status"] == true

    max_dvm, max_dva, max_dpg, max_dqg = _solution_diffs(result_trad["solution"], result_grainger["solution"])
    matches = converged_trad && converged_grainger &&
        max_dvm < SOLUTION_TOL && max_dva < SOLUTION_TOL && max_dpg < SOLUTION_TOL && max_dqg < SOLUTION_TOL

    return GraingerCaseResult(case_name, nbus, t_trad, t_grainger, converged_trad, converged_grainger,
        matches, max_dvm, max_dva, max_dpg, max_dqg, "")
end

function _print_report(results)
    println()
    println("="^104)
    println("Grainger vs. traditional AC power flow comparison")
    println("="^104)
    @printf("%-20s %6s %12s %14s %8s %10s %10s %10s %10s\n",
        "case", "nbus", "t_trad (s)", "t_grainger (s)", "match?", "max|Δvm|", "max|Δva|", "max|Δpg|", "max|Δqg|")
    for r in results
        if !isempty(r.error_msg)
            @printf("%-20s %6d %12s %14s %8s   %s\n", r.case_name, r.nbus, "ERROR", "ERROR", "-", r.error_msg)
            continue
        end
        match_str = r.solutions_match ? "yes" : "NO"
        @printf("%-20s %6d %12.4f %14.4f %8s %10.2e %10.2e %10.2e %10.2e\n",
            r.case_name, r.nbus, r.time_traditional, r.time_grainger, match_str,
            r.max_dvm, r.max_dva, r.max_dpg, r.max_dqg)
        if !r.solutions_match && (!r.converged_traditional || !r.converged_grainger)
            println("    -> did not converge (traditional converged=$(r.converged_traditional), grainger converged=$(r.converged_grainger))")
        end
    end
    println("="^104)
end

"""
Line plot of computation time by test case: one point per case per
implementation, evenly spaced along the x-axis in ascending bus-count order
(not spaced by nbus value), with a dashed line connecting all traditional
points and a separate dashed line connecting all grainger points. Saved to
PLOT_PATH.
"""
function _plot_timing(results)
    ok = sort(filter(r -> isempty(r.error_msg), results), by = r -> r.nbus)
    isempty(ok) && (@warn "no successful runs to plot"; return nothing)

    x = 1:length(ok)
    tick_labels = ["$(r.case_name)\n($(r.nbus) bus)" for r in ok]
    t_trad = [r.time_traditional for r in ok]
    t_grainger = [r.time_grainger for r in ok]

    plt = plot(
        x, t_trad;
        label = "Traditional", color = COLOR_TRADITIONAL, linewidth = 2, linestyle = :dash,
        markershape = :circle, markersize = 6, markerstrokewidth = 0,
        yscale = :log10,
        xticks = (x, tick_labels), xrotation = 45,
        xlabel = "Test case (ordered by number of buses)", ylabel = "Computation time, s (log scale)",
        title = "AC power flow computation time: traditional vs. grainger",
        legend = :topleft, gridalpha = 0.2, framestyle = :box,
        size = (1100, 650), margin = 8Plots.mm,
    )
    plot!(plt, x, t_grainger;
        label = "Grainger", color = COLOR_GRAINGER, linewidth = 2, linestyle = :dash,
        markershape = :circle, markersize = 6, markerstrokewidth = 0)

    savefig(plt, PLOT_PATH)
    return PLOT_PATH
end

"NaN/Inf aren't valid JSON; map them to null so the report file parses cleanly"
_json_safe(x::Float64) = isfinite(x) ? x : nothing

"write the per-case comparison results to REPORT_PATH as JSON"
function _write_report(results, path)
    report = [
        Dict(
            "case_name" => r.case_name,
            "nbus" => r.nbus,
            "time_traditional_s" => _json_safe(r.time_traditional),
            "time_grainger_s" => _json_safe(r.time_grainger),
            "converged_traditional" => r.converged_traditional,
            "converged_grainger" => r.converged_grainger,
            "solutions_match" => r.solutions_match,
            "max_abs_diff" => Dict(
                "vm" => _json_safe(r.max_dvm),
                "va" => _json_safe(r.max_dva),
                "pg" => _json_safe(r.max_dpg),
                "qg" => _json_safe(r.max_dqg),
            ),
            "solution_tolerance" => SOLUTION_TOL,
            "error" => isempty(r.error_msg) ? nothing : r.error_msg,
        )
        for r in results
    ]
    open(path, "w") do io
        JSON.print(io, report, 2)
    end
    return path
end

"run both code paths once on the smallest available case so Julia's JIT compilation cost lands here, not on the first timed case"
function _warmup(cases)
    isempty(cases) && return
    warmup = first(cases)
    println("Warming up JIT on $(warmup.case_name)...")
    PowerModels.compute_ac_pf(deepcopy(warmup.data); mapping = true, grainger = false, enforce_q_lims = false)
    PowerModels.compute_ac_pf(deepcopy(warmup.data); mapping = true, grainger = true, enforce_q_lims = false)
    return
end

function main()
    cases = _load_test_cases()
    println("Found $(length(cases)) test case(s) under $(NETWORK_INFO_DIR)")
    _warmup(cases)

    results = GraingerCaseResult[]
    for tc in cases
        println("\nRunning $(tc.case_name) (nbus=$(tc.nbus))...")
        try
            r = _run_case(tc.case_name, tc.data)
            push!(results, r)
            println("  t_trad=$(round(r.time_traditional, digits=4))s  " *
                    "t_grainger=$(round(r.time_grainger, digits=4))s  match=$(r.solutions_match)")
        catch e
            @warn "case $(tc.case_name) failed" exception=(e, catch_backtrace())
            push!(results, GraingerCaseResult(tc.case_name, tc.nbus, NaN, NaN, false, false, false,
                NaN, NaN, NaN, NaN, sprint(showerror, e)))
        end
    end

    _print_report(results)
    report_path = _write_report(results, REPORT_PATH)
    println("Saved comparison report to $(report_path)")
    plot_path = _plot_timing(results)
    plot_path !== nothing && println("Saved timing plot to $(plot_path)")
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
