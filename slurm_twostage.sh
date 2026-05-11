#!/bin/bash
# ============================================================
# SLURM job script for GED-PT two-stage inference
# Uses HuggingFace Transformers (no Ollama)
#
# Submit:
#   sbatch slurm_2stage.sh --model Qwen/Qwen3-8B
#   sbatch slurm_2stage.sh --model TucanoBR/Tucano-2b4-Instruct
#   sbatch slurm_2stage.sh --model google/gemma-3-12b-it
#   sbatch slurm_2stage.sh --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# ============================================================

#SBATCH --job-name=ged_2stage
#SBATCH --account=f202500017aivlabdeucaliong
#SBATCH --gpus=1
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive

# ---- parse args ----
MODEL="Qwen/Qwen3-8B"
SPLIT="test"
CONFIG="config.yaml"
BATCH_SIZE=0   # 0 = auto-detect from VRAM

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)      MODEL="$2";      shift 2 ;;
        --split)      SPLIT="$2";      shift 2 ;;
        --config)     CONFIG="$2";     shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        *)            shift ;;
    esac
done

echo "=========================================="
echo "  GED-PT Two-Stage Inference"
echo "  Model    : $MODEL"
echo "  Split    : $SPLIT"
echo "  Config   : $CONFIG"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Date     : $(date)"
echo "=========================================="

mkdir -p logs predictions metrics hf_cache

# ---- environment ----
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1  # may be needed explicitly

source /projects/F202600026AIVLABDEUCALION/evelinamorim/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="/projects/F202600026AIVLABDEUCALION/evelinamorim/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

PYTHON=$(which python3)
echo "Python: $PYTHON ($($PYTHON --version))"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# ---- install dependencies if missing ----
$PYTHON -c "import transformers" 2>/dev/null || pip install transformers accelerate --quiet
$PYTHON -c "import accelerate" 2>/dev/null || pip install accelerate --quiet

# ---- run two-stage inference ----
echo ""
echo "[$(date +%T)] Starting two-stage inference ..."
$PYTHON run_inference_2stage.py \
    --model "$MODEL" \
    --split "$SPLIT" \
    --config "$CONFIG" \
    --batch_size "$BATCH_SIZE" \
    --hf_cache "$HF_HOME"

EXIT_CODE=$?

echo ""
echo "[$(date +%T)] Done. Exit code: $EXIT_CODE"
exit $EXIT_CODE