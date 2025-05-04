#!/bin/bash
#SBATCH --job-name=transf-prog-rasp
#SBATCH --output=logs/llm_%j.out
#SBATCH --error=logs/llm_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
##SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1

# 2. Activate your prepared conda environment
source /scratch/gpfs/am9755/miniconda3/bin/activate tp

# (Optional) ensure project root is in $PYTHONPATH so "models" & "utils" import
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# 3. Run the RASP task loop
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
python -c "import torch; print('Torch sees CUDA:', torch.cuda.is_available())"
bash scripts/rasp-1.sh