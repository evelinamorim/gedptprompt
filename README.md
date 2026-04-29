# GED-PT SLM Benchmarking Pipeline

Grammar Error Detection for Brazilian Portuguese — prompting-based evaluation of
open Small Language Models via Ollama on the Deucalion supercomputer.

## File structure

```
ged_slm/
├── config.yaml          # All experiment parameters
├── data_reader.py       # BIO TSV parser → Sentence objects
├── run_inference.py     # Prompts Ollama, saves predictions JSON
├── evaluate.py          # Span-level P/R/F1 + token accuracy
├── slurm_eval.sh        # End-to-end SLURM job (inference + eval)
├── requirements.txt
└── data/
    ├── train_bio.tsv
    ├── val_bio.tsv
    └── test_bio.tsv
```

## Quick start (local / interactive)

```bash
pip install -r requirements.txt

# Check data
python data_reader.py data/test_bio.tsv

# Run inference (zero-shot, test split)
python run_inference.py --config config.yaml --split test --strategy zero_shot

# Evaluate
python evaluate.py --predictions predictions/gemma3-12b_test_zero_shot.json --verbose
```

## Running on Deucalion

```bash
# Zero-shot with Gemma3
sbatch slurm_eval.sh --model gemma3:12b --strategy zero_shot --split test

# Few-shot with Qwen2.5
sbatch slurm_eval.sh --model qwen2.5:14b --strategy few_shot --split test

# Validate on val split first
sbatch slurm_eval.sh --model qwen2.5:7b --strategy zero_shot --split val
```

Logs land in `logs/ged_slm_<JOBID>.out`.

## Metrics

Evaluation is **span-level exact match**:
- A predicted span (start, end) is a TP only if it exactly matches a gold span in the same sentence.
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1** = harmonic mean of P and R

Token-level accuracy is also reported as a supplementary signal.

## Suggested model sweep

| Model tag (Ollama)  | Size  | Notes                        |
|---------------------|-------|------------------------------|
| `gemma3:4b`         | ~3 GB | Fast baseline                |
| `gemma3:12b`        | ~8 GB | Recommended default          |
| `qwen2.5:7b`        | ~5 GB | Strong multilingual          |
| `qwen2.5:14b`       | ~9 GB | Best Qwen multilingual       |
| `llama3.1:8b`       | ~5 GB | Meta baseline                |
| `aya:35b`           | ~20GB | Multilingual-specific        |

## Changing prompting strategy

Edit `config.yaml`:
```yaml
prompting:
  strategy: "few_shot"          # zero_shot | few_shot
  num_few_shot_examples: 3
```
Or pass `--strategy few_shot` to the scripts directly.
