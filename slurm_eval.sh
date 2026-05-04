#!/bin/bash
# ============================================================
# SLURM job script for GED-PT SLM benchmarking on Deucalion
# Runs inference + evaluation for a single model/strategy combo.
#
# Submit:
#   sbatch slurm_eval.sh --model gemma3:12b --strategy zero_shot
#   sbatch slurm_eval.sh --model qwen2.5:14b --strategy few_shot --split val
# ============================================================

#SBATCH --job-name=ged_slm_gemma4_e4b
#SBATCH --account=f202500017aivlabdeucaliong
#SBATCH --gpus=1
#SBATCH --partition normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00          # wall time: adjust per model size
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=logs/%x_%j_gemma4_e4b.out
#SBATCH --error=logs/%x_%j_gemma4_e4b.err

# ---- parse optional overrides from sbatch extra args ----
MODEL="gemma4:e4b"
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
module load ollama        # Deucalion: load the Ollama module
module load python/3.11   # or whichever Python module is available

# Activate your virtualenv / conda env if needed:
# source /path/to/venv/bin/activate
# conda activate ged_slm

# ---- start Ollama server in the background ----
# Deucalion may require pointing OLLAMA_MODELS to a shared storage path
export OLLAMA_MODELS="${SCRATCH:-$HOME}/.ollama/models"
export OLLAMA_HOST="http://127.0.0.1:11434"

ollama serve &
OLLAMA_PID=$!
echo "Ollama server started (PID=$OLLAMA_PID)"

# Give the server a few seconds to initialise
sleep 10

# Pull model if not already cached (no-op if present)
#echo "Ensuring model is available: $MODEL"
#ollama pull "$MODEL"

# ---- update config with the requested model ----
# We write a temporary config to avoid mutating the shared one
TMP_CONFIG="config_${SLURM_JOB_ID}.yaml"
python3 - <<PYEOF
import yaml, sys

with open("$CONFIG") as f:
    cfg = yaml.safe_load(f)

cfg["model"]["name"] = "$MODEL"
cfg["prompting"]["strategy"] = "$STRATEGY"

with open("$TMP_CONFIG", "w") as f:
    yaml.dump(cfg, f, allow_unicode=True)

print("Temporary config written:", "$TMP_CONFIG")
PYEOF

# ---- inference ----
echo ""
echo "[$(date +%T)] Starting inference ..."
python3 run_inference.py \
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
python3 evaluate.py \
    --predictions "$PRED_FILE" \
    --verbose

EVAL_EXIT=$?

# ---- cleanup ----
kill $OLLAMA_PID 2>/dev/null
rm -f "$TMP_CONFIG"

echo ""
echo "[$(date +%T)] Done. Exit code: $EVAL_EXIT"
exit $EVAL_EXIT
