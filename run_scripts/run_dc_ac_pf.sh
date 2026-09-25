#!/bin/bash
#SBATCH --job-name=dc-ac-pf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=anva7450@colorado.edu
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --export=NONE
#SBATCH --array=0-4
#SBATCH --output=/scratch/alpine/anva7450/logs/dc_ac_pf/%x-%a-%j.out

# One array task per case, running generate_dataset.jl's dc_ac_pf pass.
# mkdir -p /scratch/alpine/anva7450/logs/dc_ac_pf once, or sbatch cannot write the logs.
#   sbatch run_dc_ac_pf.sh                                      # case14 .. case300
#   sbatch --array=5-7 --cpus-per-task=8 --mem=48G \
#          --time=23:00:00 run_dc_ac_pf.sh                       # case2869_pegase .. case9241_pegase
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

PERT_NAME=extreme_pert
DEVICE=false
# batches are held in memory before the h5 flush, so the big cases want a smaller one
BATCH_SIZE=$([[ ${SLURM_ARRAY_TASK_ID:-0} -ge 5 ]] && echo 25 || echo 100)

# ordered small -> large so an --array range picks a size tier
CASES=(
   case14 case30  case57 
   case118  case300
   case2869_pegase
   case7336  case9241_pegase
)

CASE_NAME=${CASES[${SLURM_ARRAY_TASK_ID:?run me with sbatch --array}]}
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

echo "host=$(hostname)  case=$CASE_NAME  pert=$PERT_NAME  device=$DEVICE"
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
    "$REPO/run_scripts/generate_dataset.jl" "$CASE_NAME" "$PERT_NAME" "$DEVICE" "$BATCH_SIZE"
echo "[$(date +%T)] done"
