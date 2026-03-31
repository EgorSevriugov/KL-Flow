# KL-Flow: Non-Autoregressive Text Generation with Logit-Space Flow Matching

Reference implementation for KL-geodesic flow matching in discrete sequence modeling.

This repository contains training, inference, and evaluation code used for:
- unconditional text generation,
- conditional text generation,
- code infilling.

---

## Project Overview

KL-Flow models generation as transport on the probability simplex using KL-geodesic interpolation (linear in logit space).  
The codebase is built around practical research workflows: config-driven experiments, multi-GPU training, and task-specific evaluation scripts.

### Highlights
- KL-geodesic (logit-space) flow matching for discrete tokens
- Unified training entrypoint for unconditional and conditional tasks
- Deterministic, sampling, and hybrid inference modes
- Built-in evaluation scripts for text and code generation tasks

---

## Paper and Citation

- OpenReview (ICLR 2026 submission): [Logit-KL Flow Matching: Non-Autoregressive Text Generation via Sampling-Hybrid Inference](https://openreview.net/forum?id=scgtQSpROE)
- If this repository helps your research, please cite the paper and the code.

### Main Results (from submission)

| Task | Dataset | Metric | Baseline | KL-Flow | Notes |
|---|---|---|---:|---:|---|
| Unconditional text generation | FineFineWeb | GPT-2 perplexity (lower is better) | 97.2 (GPT-2, NFE=1024) | 62.9 (KL-Flow 150M, NFE=1024) | Better quality at matched evaluation setup |
| Conditional generation | WMT14 De-En | BLEU Top-5 (higher is better) | 21.3 (DFM) | 27.0 (KL-Flow sampling) | Strong gain on translation benchmark |
| Conditional generation | Lamini Instruction | BLEU Top-5 (higher is better) | 8.1 (DFM) | 9.5 (KL-Flow hybrid) | Best result among compared NAR methods |
| Code infilling | MBPP (10% lines removed) | Pass@1 (higher is better) | 11.1 (DFM) | 17.4 (KL-Flow) | Large improvement in functional correctness |

---

## Repository Map

- `train_fm.py` - main training script
- `download_dataset.py` - dataset download/prep helper
- `inference_fm.py` - unconditional/conditional inference
- `inference_fm_infilling.py` - code infilling inference
- `eval_unconditional.py` - unconditional generation evaluation
- `eval_conditional.py` - conditional generation evaluation
- `eval_infilling.py` - code infilling evaluation
- `configs/` - experiment configs
- `models/` - model implementations

---

## Quick Start (UV)

### 1) Create environment and install dependencies

```bash
uv venv
uv pip install -r requirements.txt
```

### 2) Download dataset (TinyStories example)

```bash
uv run python download_dataset.py configs/config_tinystories_unconditional.yaml
```

### 3) Train

Single GPU:
```bash
uv run python train_fm.py configs/config_tinystories_unconditional.yaml
```

Multi-GPU:
```bash
uv run torchrun --nproc_per_node=8 train_fm.py configs/config_tinystories_unconditional.yaml
```

---

## Inference

Unconditional / conditional:

```bash
uv run python inference_fm.py configs/config_tinystories_unconditional.yaml
```

Code infilling:

```bash
uv run python inference_fm_infilling.py configs/config_tinystories_unconditional.yaml POSTFIX
```

Outputs are saved under `logs/{run_name}/inference/`.

---

## Evaluation

Unconditional:
```bash
uv run python eval_unconditional.py configs/config_tinystories_unconditional.yaml
```

Conditional:
```bash
uv run python eval_conditional.py configs/config_conditional_alpaca.yaml
```

Infilling:
```bash
uv run python eval_infilling.py configs/config_tinystories_unconditional.yaml
```

Evaluation artifacts are saved as `logs/{run_name}/eval_*.pt`.

---

## Minimal Config You Usually Edit

In most experiments, only these fields are essential:

```yaml
training_config:
  run_name: "tinystories_klflow"

data:
  dataset_path: "./data/TinyStories"   # local path or HF dataset id
  tokenizer_name: "gpt2"
  sequence_length: 512                 # single source of truth across train/infer/eval/model
  condition: false

fm:
  type: "Logit"                        # recommended default
```

Use `configs/config_tinystories_unconditional.yaml` as the base template.

---

## Hardware Notes

- Python 3.8+
- CUDA-capable GPU recommended
- For large runs, multi-GPU setup is expected

---

## Citation

If you use this repository in academic work, please cite:
- the paper on OpenReview: https://openreview.net/forum?id=scgtQSpROE
- this code repository

```bibtex
@inproceedings{sevriugov2026logitklflow,
  title     = {Logit-KL Flow Matching: Non-Autoregressive Text Generation via Sampling-Hybrid Inference},
  author    = {Sevriugov, Egor and Dragunov, Nikita and Razzhigaev, Anton and Kuznetsov, Andrey and Oseledets, Ivan},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=scgtQSpROE},
  note      = {OpenReview: scgtQSpROE}
}
```
