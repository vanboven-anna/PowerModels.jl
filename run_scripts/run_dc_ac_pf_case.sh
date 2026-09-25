#!/bin/bash
#SBATCH --job-name=dc-ac-pf-9241
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=23:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=anva7450@colorado.edu
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --export=NONE
#SBATCH --array=0-7
#SBATCH --output=/scratch/alpine/anva7450/logs/dc_ac_pf/%x-%a-%j.out

# Shards case9241_pegase's remaining datapoints across NUM_SHARDS concurrent
# tasks (Ipopt/MUMPS is single-core per solve, so this is how more resources
# actually buys speed -- see generate_dataset.jl's generate_optimal_dataset).
# Only submit once any earlier, unsharded case9241 job has stopped: both
# write dc_ac_pf/dataset.h5, and two writers on it will corrupt it.
#
#   sbatch run_dc_ac_pf_case.sh
#   # once every array task shows COMPLETED in squeue/sacct:
#   julia --project=/home/anva7450/bus_swap/PowerModels.jl \
#     /home/anva7450/bus_swap/PowerModels.jl/run_scripts/generate_dataset.jl \
#     merge case9241_pegase extreme_pert false 8
#
# NUM_SHARDS below must equal (highest --array index) + 1.
set -euo pipefail

REPO=/home/anva7450/bus_swap/PowerModels.jl
DATA_ROOT=/projects/anva7450/bus_swap/bus_swap_data

# --export=NONE leaves lmod uninitialized; its init script trips over set -u
set +u
source /curc/sw/lmod/lmod/init/bash
module purge
module load julia/1.11.6
set -u
JULIA=$(command -v julia)

# after the module load, which exports a JULIA_DEPOT_PATH of its own
export JULIA_DEPOT_PATH=/projects/anva7450/julia_depot

CASE_NAME=case9241_pegase
PERT_NAME=extreme_pert
DEVICE=false
BATCH_SIZE=25
NUM_SHARDS=8

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:?run me with sbatch --array}
DATASET_DIR=$DATA_ROOT/test_cases/data/$CASE_NAME/$PERT_NAME

# Ipopt/MUMPS is serial; extra BLAS threads only oversubscribe the allocation
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JULIA_NUM_THREADS=1
export JULIA_NUM_GC_THREADS=${SLURM_CPUS_PER_TASK:-4}
export JULIA_PROJECT=$REPO

# this task's own log, per the --output pattern; dropped unless the run succeeds
LOG=/scratch/alpine/anva7450/logs/dc_ac_pf/${SLURM_JOB_NAME}-${SLURM_ARRAY_TASK_ID}-${SLURM_JOB_ID}.out
cleanup() { local rc=$?; [[ $rc -eq 0 ]] || rm -f "$LOG"; return 0; }
trap cleanup EXIT

echo "host=$(hostname)  case=$CASE_NAME  shard=$SHARD_INDEX/$NUM_SHARDS"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}M  depot=$JULIA_DEPOT_PATH"

[[ -n "$JULIA" && -x "$JULIA" ]] || { echo "ERROR: module load julia/1.11.6 did not put julia on PATH"; exit 1; }
[[ -d "$DATASET_DIR/baseline_acpf" ]] || { echo "ERROR: no baseline_acpf under $DATASET_DIR"; exit 1; }
[[ -f "$REPO/Project.toml" ]] || { echo "ERROR: $REPO is not the PowerModels.jl repo root"; exit 1; }
[[ -f "$REPO/config.jl" ]] || { echo "ERROR: no config.jl in $REPO -- copy sample_config.jl there"; exit 1; }
grep -q "$DATA_ROOT" "$REPO/config.jl" || {
  echo "ERROR: $REPO/config.jl does not point at $DATA_ROOT. It should read:"
  echo "  const TESTCASE_PATH = abspath(\"$DATA_ROOT/test_cases\")"
  echo "  const RESULTS_PATH  = abspath(\"$DATA_ROOT/results\")"
  echo "  const DATA_PATH     = abspath(\"$DATA_ROOT\")"
  exit 1
}

# first task in serializes the depot build; the rest find it current and fall through
mkdir -p "$JULIA_DEPOT_PATH"
echo "[$(date +%T)] waiting for precompile lock..."
(
  flock 9
  echo "[$(date +%T)] holding lock, instantiating/precompiling"
  "$JULIA" --startup-file=no --project="$REPO" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
) 9>"$JULIA_DEPOT_PATH/dcacpf_precompile.lock"
echo "[$(date +%T)] depot ready, starting solve"

"$JULIA" --startup-file=no --project="$REPO" \
    "$REPO/run_scripts/generate_dataset.jl" \
    "$CASE_NAME" "$PERT_NAME" "$DEVICE" "$BATCH_SIZE" "$SHARD_INDEX" "$NUM_SHARDS"
echo "[$(date +%T)] done"
