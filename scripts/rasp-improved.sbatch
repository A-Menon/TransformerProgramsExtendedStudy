#!/bin/bash
#SBATCH --job-name=tp-rasp
#SBATCH --array=1-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# list of single improvements + "all"
IMPROV_LIST=(
  "prefixsum,sketch,chunks,experts,contrast,memory"
)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
IMPROV="${IMPROV_LIST[$IDX]}"

# sanitize for filenames: commas → +
SAFE_IMP=$(echo "$IMPROV" | tr ',' '+')

LOGDIR="logs-improved/${SAFE_IMP}"
mkdir -p "$LOGDIR"

echo "=== TASK ${SLURM_ARRAY_TASK_ID}: improvements='$IMPROV' ==="

# redirect entire script’s logs
exec >"${LOGDIR}/run_${SAFE_IMP}.out" 2>"${LOGDIR}/run_${SAFE_IMP}.err"

# load env
source /scratch/gpfs/am9755/miniconda3/bin/activate tp
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# sanity check
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python - <<EOF
import torch
print('Torch CUDA available:', torch.cuda.is_available())
EOF

# run the loop with this improvement set
bash scripts/rasp-improved.sh "$IMPROV"
