# MiniCPM5-1B + Frozen SigLIP ViT — Flamingo-style Gated Cross-Attention for Agricultural VQA

Generative visual question answering on **PlantVillageVQA**. A frozen MiniCPM5-1B
(1.5B) language model is augmented with **5 gated cross-attention blocks** (Flamingo
style) at decoder layers **7/10/13/16/19**, each fed by a per-layer projector from a
**frozen SigLIP ViT-B/16**. Only the projectors and cross-attention blocks train
(~159M params); both backbones stay frozen. Image features are precomputed and cached.

Code is authored locally; **training runs on Kaggle Notebooks (2× T4, fp16)**.

## Layout
```
configs/default.yaml     # all hyperparameters
data/prepare.py          # load CSV, join images, derive complexity, stratified subset, split-by-image
data/cache_features.py   # precompute frozen SigLIP [196,768] features per image
data/dataset.py          # sequence layout ([BOS] Q A [EOS]), answer-only loss mask, collate
models/vision.py         # frozen SigLIP wrapper
models/projector.py      # per-layer projector (+ optional Perceiver resampler)
models/xattn.py          # GatedCrossAttn (zero-init tanh gates)
models/model.py          # wraps MiniCPM decoder layers; trainable-param reporter
train.py                 # fp16 autocast + GradScaler, grad checkpointing, cosine+warmup
eval.py                  # greedy/beam generation; EM / token-F1 / per-category & per-complexity
analyze_gates.py         # gate-magnitude bar chart for the model card
kaggle_notebook.ipynb    # thin driver for Kaggle
```

## Pipeline
```bash
# 1. prepare a stratified 30k-QA subset, split by image (no image crosses splits)
python -m data.prepare --config configs/default.yaml --subset-size 30000
# 2. cache frozen SigLIP features (one .pt per image)
python -m data.cache_features --config configs/default.yaml
# 3. train (single GPU)
python train.py --config configs/default.yaml
# 4. evaluate + baselines
python eval.py --config configs/default.yaml --ckpt runs/exp1/best.pt --split test
python eval.py --config configs/default.yaml --ckpt runs/exp1/best.pt --split test --blind   # blind-LLM baseline
# 5. gate figure
python analyze_gates.py --ckpt runs/exp1/best.pt --out runs/exp1/gates.png
```

Sanity check (build step 5 of the plan) — overfit a tiny slice; loss should fall and
EM rise, confirming the loss mask and gate gradients:
```bash
python train.py --overfit 40 --grad-accum 1 --epochs 60
```

## Data notes (verified against the real dataset)
- The HF repo ships a single `PlantVillageVQA.zip` (831 MB). `load_dataset(...)` misreads
  it as an image folder — **do not** use the auto-loader. On Kaggle, add the dataset as a
  Notebook input (it mounts unzipped) and point `data.data_root` at it.
- CSV schema: `image_id, question_type, question, answer, image_path, split`.
  193,609 QA / 55,448 images / 9 question types. Splits are `train`/`test` only; we carve
  `val` (and an extra held-out test) out of train **by image**.
- `image_path` in the CSV does not match the on-disk layout and image extensions mix
  `.JPG`/`.jpg`; we join on `image_id` stem, case-insensitively.
- No `complexity` column — we derive a 3-level complexity from the 9 `question_type`
  categories (see `data/prepare.py:COMPLEXITY_MAP`).
- Answers are short (1–18 words, mean 5.2); ~42% are yes/no → exact-match is the primary
  metric and the blind-LLM baseline exposes the language prior.

## Hardware constraints (Kaggle 2× T4, Turing)
- **fp16 only** (no bf16 on T4): autocast(fp16) + GradScaler; LayerNorms/gates run in fp32.
- **No FlashAttention**: `attn_implementation="sdpa"`. Do not install `flash-attn`.
- Gradient checkpointing on the LLM; micro-batch 2 + grad-accum 16; cached image features.
- Single-GPU is the default and validated path; DDP is optional throughput-only.

## Citation / License
PlantVillageVQA — Sakib et al. 2025, arXiv:2508.17117; PlantVillage source corpus.
Dataset license **CC BY 4.0**.
