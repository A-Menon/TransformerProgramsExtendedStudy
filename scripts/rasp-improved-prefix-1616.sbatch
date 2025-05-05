#!/bin/bash
#SBATCH --job-name=transf-prog-rasp
#SBATCH --output=logs-improved/rasp-improved-prefix-1616.out
#SBATCH --error=logs-improved/rasp-improved-prefix-1616.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
##SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1

# 2. Activate your prepared conda environment
source /scratch/gpfs/am9755/miniconda3/bin/activate tp

# (Optional) ensure project root is in $PYTHONPATH so "models" & "utils" import
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# 3. Run the RASP task loop
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('Torch sees CUDA:', torch.cuda.is_available())"
bash scripts/rasp-improved-prefix-1616.sh