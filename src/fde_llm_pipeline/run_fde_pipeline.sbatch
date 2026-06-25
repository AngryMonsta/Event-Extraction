#!/bin/bash
# =============================================================
# WWF 2026 — Gemma 4 FDE ETL Job
# Dataset root: /storage/home/hcoda1/1/czhang808/scratch/WWF/proto_data
# Model:        google/gemma-4-E4B-it
# Submit:       sbatch run_fde_pipeline.sbatch  (from fde_llm_pipeline dir)
# =============================================================
#SBATCH --job-name=fdePipeline
#SBATCH --account=paceship-clef2026_img_reasoning
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -q embers
#SBATCH --output=../outputs/fde_pipeline/p_%j.log
#SBATCH --error=../outputs/fde_pipeline/p_%j.err

set -euo pipefail

# ── Job info ─────────────────────────────────────────────────
echo "=========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "GPU      : $CUDA_VISIBLE_DEVICES"
echo "Started  : $(date)"
echo "=========================================="

# ── Conda ────────────────────────────────────────────────────
module load anaconda3/2023.03

source /storage/home/hcoda1/1/czhang808/scratch/WWF/.venv/bin/activate


# ── Working directory ────────────────────────────────────────
USER_ROOT="/storage/home/hcoda1/1/czhang808/scratch/WWF"
cd "$USER_ROOT/fde_llm_pipeline" || exit 1
echo "Working dir: $(pwd)"

export PYTHONPATH="$USER_ROOT/fde_llm_pipeline:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1

# ── Hugging Face cache ───────────────────────────────────────
# Reuse the shared scratch cache instead of downloading into home storage.
export HF_HOME="/storage/home/hcoda1/1/czhang808/scratch/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
mkdir -p "$HF_HUB_CACHE"

# If a token was saved by `hf auth login`, export it for this non-interactive job.
if [[ -f "$HF_HOME/token" ]]; then
    export HF_TOKEN="$(< "$HF_HOME/token")"
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

# ── Script & data paths ──────────────────────────────────────
SCRIPT="$USER_ROOT/fde_llm_pipeline/run_pipeline.py"
DATA_FOLDER="$USER_ROOT/data/test_data"

# ── Pre-flight checks ────────────────────────────────────────
mkdir -p "$USER_ROOT/outputs/fde_pipeline"

echo "Python  : $(which python)"
python -c "import torch; print('torch :', torch.__version__); print('cuda  :', torch.cuda.is_available())"

# ── Run ──────────────────────────────────────────────────────
# Output JSON is auto-named inside proto_data/DisputeDetect/
# Full run over all articles:
python "$SCRIPT" \
    --zip       "$DATA_FOLDER/test_raw.ZIP" \
    --output    "$USER_ROOT/results" \
    --model     google/gemma-4-E4B-it \
    --checkpoint-every 10 \


echo "=========================================="
echo "Finished : $(date)"
echo "=========================================="