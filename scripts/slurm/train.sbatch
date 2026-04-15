#!/bin/bash
# Generic slurm wrapper around scripts/train.py for the Euler cluster.
#
# Usage
# -----
#   sbatch scripts/slurm/train.sbatch task=spiking-speech-commands
#   sbatch scripts/slurm/train.sbatch task=shd-classification training.num_epochs=60
#
# All arguments after the script name are passed straight through to Hydra.
#
# Customize the resource block below per dataset if needed. Defaults below
# fit the S7 paper configs comfortably on 2× RTX 3090 for up to 20 h of
# training.
#
#SBATCH --job-name=s7-train
#SBATCH --partition=gpuhe.24h
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=rtx_3090:2
#SBATCH --mem-per-cpu=16G
#SBATCH --output=/cluster/scratch/haihao/claude/slurm-logs/%x-%j.out

set -euo pipefail

# Conda
source /cluster/home/haihao/miniconda3/etc/profile.d/conda.sh
conda activate s7

# Repo root
cd /cluster/home/haihao/haiqing/S7
export PYTHONPATH=.

# Let JAX grab most of each GPU (DVS128 needs this; harmless for smaller tasks).
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Redirect HF datasets / transformers / hub caches to scratch — home dir is
# only 50 GB and AAN's pre-tokenized arrow shards alone can blow past that.
export HF_HOME=/cluster/scratch/haihao/claude/hf_cache
export HF_DATASETS_CACHE=/cluster/scratch/haihao/claude/hf_cache/datasets

echo "[slurm] node=$(hostname) job=${SLURM_JOB_ID:-?} gpus=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-?}}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# Default to num_workers=2 (memcg-safe) and wandb off unless overridden.
python scripts/train.py \
    training.num_workers=2 \
    logging.wandb=false \
    "$@"
