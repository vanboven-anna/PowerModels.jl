
using Pkg

# Known failing runs from past sweeps, keyed by a short label you can pass as
# the sole CLI arg to re-run just that one (case, datapoint, config) combo.
const BUSSWAP_KNOWN_FAILURES = Dict(
    # errored with InterruptException during the full 66-run sweep on 2026-08-24
    "rerun-failed" => (case = "case300", datapoint = "14", config = "nearest_gen+no_grainger+no_obo"),
)

# When this file is executed directly as a script, activate the local project
# environment and run the standalone bus-swap test with progress reporting on.
# Positional CLI args (all optional, comma-separated, default = unfiltered = all):
#   ARGS[1] -> PM_BUSSWAP_CASES
#   ARGS[2] -> PM_BUSSWAP_DATAPOINTS
#   ARGS[3] -> PM_BUSSWAP_CONFIGS
# A single arg matching a key in BUSSWAP_KNOWN_FAILURES overrides all three
# with that run's exact case/datapoint/config.
if abspath(PROGRAM_FILE) == @__FILE__
    repo_root = normpath(joinpath(@__DIR__, ".."))
    Pkg.activate(repo_root)
    ENV["PM_BUSSWAP_PROGRESS"] = "1"

    if length(ARGS) == 1 && haskey(BUSSWAP_KNOWN_FAILURES, ARGS[1])
        known = BUSSWAP_KNOWN_FAILURES[ARGS[1]]
        ENV["PM_BUSSWAP_CASES"] = known.case
        ENV["PM_BUSSWAP_DATAPOINTS"] = known.datapoint
        ENV["PM_BUSSWAP_CONFIGS"] = known.config
    else
        length(ARGS) >= 1 && !isempty(ARGS[1]) && (ENV["PM_BUSSWAP_CASES"] = ARGS[1])
        length(ARGS) >= 2 && !isempty(ARGS[2]) && (ENV["PM_BUSSWAP_DATAPOINTS"] = ARGS[2])
        length(ARGS) >= 3 && !isempty(ARGS[3]) && (ENV["PM_BUSSWAP_CONFIGS"] = ARGS[3])
    end

    using Test
    using PowerModels
    using JSON
    include(joinpath(repo_root, "test", "pf_busswap.jl"))
end
