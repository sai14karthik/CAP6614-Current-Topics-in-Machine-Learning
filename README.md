# CAP6614 — Current Topics in Machine Learning (Spring 2026)

## Overview

This project implements and experiments with **SparseGPT**, a one-shot pruning method for large language models (LLMs).

This repo follows the project angle from the Efficient AI paper list:

> **Apply SparseGPT to prune LLaMA-2-7B at various sparsity levels, benchmark perplexity and zero-shot accuracy, and combine with quantization (sparse + quantized).**  
> Code: [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt)

Upstream SparseGPT lives in [`sparsegpt/`](sparsegpt/) (submodule / vendored copy). Project-specific evaluation scripts are in [`scripts/`](scripts/).

---

## Setup

```bash
cd /path/to/CTML_Project
python3.11 -m venv .venv   
source .venv/bin/activate
pip install -r requirements.txt
```

Put a Hugging Face token in **`.env`** at the repo root:

```bash
HF_TOKEN=hf_...
```

Required for `meta-llama/Llama-2-7b-hf` and datasets.

---

## 1. Pruning (run from `sparsegpt/`)

Calibration uses **C4** (as in the paper). Example sparsities: `0.25`, `0.5`, `0.6`.

**Sparse only**

```bash
cd sparsegpt
source ../.venv/bin/activate
python llama.py meta-llama/Llama-2-7b-hf c4 --sparsity 0.5 \
  --save ../checkpoints/llama2-7b-sparsegpt-s0.5 --skip-builtin-eval
```

**Sparse + quantized (joint, e.g. 50% sparsity + 4-bit weights)**

Uses `--wbits` in the vendored `llama.py` (same idea as the paper’s sparse + GPTQ-style quantization).

```bash
python llama.py meta-llama/Llama-2-7b-hf c4 --sparsity 0.5 --wbits 4 \
  --save ../checkpoints/llama2-7b-sparsegpt-s0.5-w4 --skip-builtin-eval
```

Repeat with other `--sparsity` / save paths for a full table. Pruning time and GPU memory depend on your hardware.

---

## 2. Benchmarks (run from repo root)

Use the **same** raw WikiText-2 test protocol and lm-eval tasks for every checkpoint (dense, sparse-only, sparse+quant).

### Perplexity (raw WikiText-2 test, full split by default)

```bash
source .venv/bin/activate
python scripts/eval_ppl_wikitext2.py \
  --model /path/to/checkpoint \
  --tokenizer meta-llama/Llama-2-7b-hf \
  --out results/ppl_<name>.json
```

### Zero-shot accuracy (lm-eval)

```bash
python scripts/eval_zeroshot.py \
  --model /path/to/checkpoint \
  --tokenizer meta-llama/Llama-2-7b-hf \
  --tasks "piqa,arc_easy,arc_challenge,lambada_openai" \
  --num-fewshot 0 \
  --out results/zeroshot_<name>.json
```

**Dense baseline:** point `--model` at `meta-llama/Llama-2-7b-hf` (no local checkpoint).

---

## 3. What to report

| Setting | Pruning command | `eval_ppl_wikitext2.py` | `eval_zeroshot.py` |
|--------|-----------------|-------------------------|---------------------|
| Dense | — | ✓ | ✓ |
| Sparse @ s | `llama.py ... --sparsity s` | ✓ | ✓ |
| Sparse @ s + 4-bit | `llama.py ... --sparsity s --wbits 4` | ✓ | ✓ |

Store JSON under [`results/`](results/) and compare perplexity + task accuracies vs dense.

---

## References

- Frantar, E., & Alistarh, D. (2023). *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* [arXiv:2301.00774](https://arxiv.org/abs/2301.00774)
- Official code: [github.com/IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt)
