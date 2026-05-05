#!/bin/bash
# ============================================================
# SLURM job script for GED-PT SLM benchmarking on Deucalion
# Runs inference + evaluation for a single model/strategy combo.
#
# Submit:
#   sbatch slurm_eval.sh --model gemma4:e4b --strategy zero_shot
#   sbatch slurm_eval.sh --model qwen3:8b --strategy zero_shot --split test
#   sbatch slurm_eval.sh --model gemma3:12b --strategy zero_shot  (use --time=08:00:00)
# ============================================================

#SBATCH --job-name=ged_slm_gemma4_e2b
#SBATCH --account=f202500017aivlabdeucaliong
#SBATCH --exclusive
#SBATCH --gpus=1
#SBATCH --partition=normal-a100-80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=logs/%x_%j_gemma4_e2b.out
#SBATCH --error=logs/%x_%j_qwen3_e2b.err

# ---- parse optional overrides from sbatch extra args ----
MODEL="gemma4:e2b"
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
echo "  GED-PT SLM Benchmark"
echo "  Model    : $MODEL"
echo "  Strategy : $STRATEGY"
echo "  Split    : $SPLIT"
echo "  Config   : $CONFIG"
echo "  Job ID   : $SLURM_JOB_ID"
echo "  Node     : $SLURMD_NODENAME"
echo "  Date     : $(date)"
echo "=========================================="

# ---- environment ----
mkdir -p logs predictions metrics

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load ollama/0.20.3-GCCcore-14.2.0-CUDA-12.8.0

source /projects/F202600026AIVLABDEUCALION/evelinamorim/venv/bin/activate

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HOME="$(pwd)/hf_cache"
export OLLAMA_MODELS=/projects/F202600026AIVLABDEUCALION/evelinamorim/ollama_models/

# Capture the correct python binary after module load
PYTHON=$(which python3)
echo "Python binary: $PYTHON"
$PYTHON --version

# Try to connect to existing Ollama instance first
if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama already running, reusing existing instance."
    OLLAMA_PID=""
else
    echo "Starting new Ollama instance..."
    ollama serve &
    OLLAMA_PID=$!
    # Wait until ready
    for i in $(seq 1 30); do
        curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1 && echo "Ollama ready." && break
        sleep 2
    done
fi

# Pre-warm: force model weights into GPU VRAM before inference starts
echo "Pre-warming model: $MODEL"
curl -s -X POST http://127.0.0.1:11434/api/chat \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Olá\"}], \"stream\": false}" \
    > /dev/null 2>&1
echo "Model warmed up."

# ---- update config with the requested model ----
# Write a temporary config per job to avoid conflicts between parallel jobs
TMP_CONFIG="config_${SLURM_JOB_ID}.yaml"
$PYTHON - <<PYEOF
import yaml, sys

with open("$CONFIG") as f:
    cfg = yaml.safe_load(f)

cfg["model"]["name"] = "$MODEL"
cfg["prompting"]["strategy"] = "$STRATEGY"
cfg["model"]["ollama_host"] = "http://127.0.0.1:11434"

with open("$TMP_CONFIG", "w") as f:
    yaml.dump(cfg, f, allow_unicode=True)

print("Temporary config written:", "$TMP_CONFIG")
PYEOF

# ---- inference ----
echo ""
echo "[$(date +%T)] Starting inference ..."
$PYTHON run_inference.py \
    --config "$TMP_CONFIG" \
    --split "$SPLIT" \
    --strategy "$STRATEGY"
INFERENCE_EXIT=$?

if [ $INFERENCE_EXIT -ne 0 ]; then
    echo "ERROR: Inference failed (exit code $INFERENCE_EXIT)"
    kill $OLLAMA_PID 2>/dev/null
    rm -f "$TMP_CONFIG"
    exit $INFERENCE_EXIT
fi

# ---- evaluation ----
MODEL_TAG=$(echo "$MODEL" | tr ':/' '--')
PRED_FILE="predictions/${MODEL_TAG}_${SPLIT}_${STRATEGY}.json"

echo ""
echo "[$(date +%T)] Running evaluation on: $PRED_FILE"
$PYTHON evaluate.py \
    --predictions "$PRED_FILE" \
    --verbose

EVAL_EXIT=$?

# ---- cleanup ----
# Only kill Ollama if this job started it
if [ -n "$OLLAMA_PID" ]; then
    kill $OLLAMA_PID 2>/dev/null
fi
rm -f "$TMP_CONFIG"

echo ""
echo "[$(date +%T)] Done. Exit code: $EVAL_EXIT"
exit $EVAL_EXIT
