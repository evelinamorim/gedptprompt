#!/bin/bash
# ============================================================
# SLURM job script for GED-PT SLM benchmarking on Deucalion
# Uses HuggingFace Transformers (replaces Ollama)
#
# Submit:
#   sbatch --job-name=gemma3_zero slurm_eval.sh --model google/gemma-3-12b-it --strategy zero_shot
#   sbatch --job-name=qwen3_zero  slurm_eval.sh --model Qwen/Qwen3-8B --strategy zero_shot
#   sbatch --job-name=tucano_zero slurm_eval.sh --model TucanoBR/Tucano-2b4-Instruct --strategy zero_shot
#   sbatch --job-name=tucano_few  slurm_eval.sh --model TucanoBR/Tucano-2b4-Instruct --strategy few_shot
# ============================================================

#SBATCH --job-name=ged_slm
#SBATCH --account=f202500017aivlabdeucaliong
#SBATCH --exclusive
#SBATCH --gpus=1
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---- parse optional overrides ----
MODEL="Qwen/Qwen3-8B"
STRATEGY="zero_shot"
SPLIT="test"
CONFIG="config.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2";    shift 2 ;;
        --strategy) STRATEGY="$2"; shift 2 ;;
        --split)    SPLIT="$2";    shift 2 ;;
        --config)   CONFIG="$2";   shift 2 ;;
        *)          shift ;;
    esac
done

echo "=========================================="
echo " GED-PT SLM Benchmark (Transformers)""
echo "  Model    : $MODEL"
echo "  Split    : $SPLIT"
echo "  Config   : $CONFIG"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Date     : $(date)"
echo "=========================================="

mkdir -p logs predictions metrics

# ---- environment ----
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1

source /projects/F202600026AIVLABDEUCALION/evelinamorim/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="/projects/F202600026AIVLABDEUCALION/evelinamorim/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"


PYTHON=$(which python3)
echo "Python: $PYTHON ($($PYTHON --version))"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"


# ---- run two-stage inference ----
echo ""
echo "[$(date +%T)] Starting two-stage inference ..."
$PYTHON run_inference_2stage.py \
    --model "$MODEL" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --hf_cache "$HF_HOME"

EXIT_CODE=$?



echo ""
echo "[$(date +%T)] Done. Exit code: $EXIT_CODE"
exit $EXIT_CODE