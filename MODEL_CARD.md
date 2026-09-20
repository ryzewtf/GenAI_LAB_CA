---
license: cc-by-4.0
datasets:
  - SyedNazmusSakib/PlantVillageVQA
language:
  - en
tags:
  - visual-question-answering
  - multimodal
  - agriculture
  - plant-disease
  - flamingo
  - gated-cross-attention
  - frozen-backbone
base_model:
  - openbmb/MiniCPM5-1B
  - google/siglip-base-patch16-224
pipeline_tag: visual-question-answering
---

# AgriVQA — Gated Cross-Attention over a Frozen MiniCPM5-1B + SigLIP

Generative visual question answering for agricultural leaf-disease images. A **frozen**
1.5B-parameter MiniCPM5-1B language model is given sight by inserting **5 Flamingo-style
gated cross-attention blocks** into its decoder, each fed by a per-layer projection of a
**frozen SigLIP ViT-B/16**. Both backbones stay frozen; only the cross-attention blocks
and projectors train, **159,383,050 parameters (~9.6 % of the ~1.66B total)**.

This card follows Mitchell et al., *Model Cards for Model Reporting* (2019).

---

## 1. Purpose

This model gives an existing, **frozen** language model the ability to answer questions about
agricultural leaf-disease images while keeping both backbones frozen. It targets
generative VQA over PlantVillage-style leaf photographs: species identification, health
assessment, attribute verification, and (with more difficulty) disease and causal reasoning.
It is a research and education demonstration of adapter-based multimodal coupling.

Three design commitments follow from that goal:

1. **Frozen backbones, adapter-only training.** The LLM already carries strong language
   priors and the ViT strong visual features. Training only a thin coupling keeps the
   parameter budget small (~9.6 %), avoids catastrophic forgetting, and fits Kaggle's 2× T4
   memory. This is the Flamingo (Alayrac et al., 2022) recipe.
2. **Cross-attention, not token concatenation.** The image never enters the token sequence.
   Text tokens *query* image tokens through dedicated cross-attention, so the LLM's
   self-attention, RoPE, and KV-cache are untouched and the 196 image tokens do not inflate
   sequence length.
3. **Zero-initialised gates.** Each new block is wrapped in a `tanh` gate initialised to 0,
   so at step 0 the augmented model is *bit-identical* to the frozen LLM. Training gradually
   "opens" the gates, letting the model set how much vision to admit at each depth.

### Model details

| | |
|---|---|
| **Architecture** | Frozen decoder-only LLM + frozen ViT, coupled by gated cross-attention |
| **Language backbone** | `openbmb/MiniCPM5-1B`: 24 decoder layers, hidden 1536, `LlamaForCausalLM`-compatible (loaded with `trust_remote_code=True`) |
| **Vision backbone** | `google/siglip-base-patch16-224`: 224px, 16×16 patches → **196 patch tokens × 768 dim** |
| **New modules** | 5 gated cross-attention blocks + 5 per-layer projectors at decoder layers **[7, 10, 13, 16, 19]** |
| **Trainable params** | **159,383,050** (projectors 17,725,440 + cross-attention 141,657,610) |
| **Frozen params** | MiniCPM5-1B (~1.5B) + SigLIP ViT-B/16 (~0.09B) |
| **Precision** | fp16 (autocast + GradScaler); LayerNorms and gates kept in fp32 |
| **Attention backend** | PyTorch SDPA (no FlashAttention; Turing/T4 target) |
| **Task** | Generative VQA (next-token, answer-span supervised) |
| **License** | CC BY 4.0 (inherited from PlantVillageVQA) |

---

## 2. Dataset

**PlantVillageVQA** (Sakib et al., 2025, arXiv:2508.17117), built on the PlantVillage leaf
corpus. CC BY 4.0.

- **Full set:** 193,609 QA pairs over 55,448 images, 14 crops / 38 diseases, 9 question types.
- **Schema (verified from the raw CSV, not assumed):**
  `image_id, question_type, question, answer, image_path, split`. There is **no** native
  `complexity` field and **no** native `val` split, only `train`/`test`.
- **Answers are short:** mean **5.17** words, **median 1**, range 1–18; **42.1 %** are
  yes/no. Because the label distribution carries a strong language prior, normalized
  exact-match is the primary metric and a blind-LLM baseline is essential.

### Derived 3-level complexity

The 9 question types are mapped to a 3-level complexity taxonomy (the dataset ships no such
field):

| Complexity | Question types |
|---|---|
| **L1, perception** | Existence & Sanity Check, Plant Species Identification, General Health Assessment |
| **L2, grounding** | Visual Attribute Grounding, Detailed Verification |
| **L3, reasoning** | Specific Disease Identification, Comprehensive Description, Causal Reasoning, Counterfactual Reasoning |

### 2.6 Data pipeline

1. **Load & join.** The HF auto-loader mis-reads the shipped `PlantVillageVQA.zip` as an
   image folder, so the CSV is read directly. The CSV `image_path` does **not** match the
   on-disk layout and extensions mix `.JPG`/`.jpg`; images are joined on the `image_id`
   **stem, case-insensitively**.
2. **Stratified subset.** A CLI-configurable subset (**default 30,000 QA pairs**) is sampled
   to preserve the per–question-type distribution, capped at ≤15,000 unique images to keep
   the feature cache small.
3. **Split by image, not by QA pair.** val (10 %) and an extra held-out test are carved out
   of the native `train` images *by image*; the native `test` images are kept as test. An
   assertion guarantees **no image appears in two splits** (multiple questions share an
   image, so a QA-level split would leak).
   - This run: **19,362 train / 2,432 val / 8,206 test QA pairs** over **12,277 unique images**.
4. **Feature cache.** The frozen ViT output is constant, so SigLIP is run **once** over every
   unique image and the `[196, 768]` fp16 patch tensor is cached to disk (one `.pt` per
   image, ≈300 KB). Training and eval then skip the ViT forward entirely, which saves VRAM
   and time. (~300 KB × 12,277 ≈ 3.6 GB.)
5. **Collate.** Question+answer sequences are padded to the batch max, cached image features
   are stacked, and answer-only label masks are built.

---

## 3. Training method

Authored locally; the full run executed on **Kaggle Notebooks (2× T4, fp16)**. Single-GPU is
the validated default (the trainable footprint is tiny; DDP would add throughput, not fit).

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW on trainable params only |
| Learning rate | 1e-4, cosine decay, 3 % warmup |
| Weight decay | 0.01 (none on gates $\alpha,\beta$, LayerNorms, biases) |
| Micro-batch / grad-accum | 2 / 16 → effective batch 32 |
| Epochs | 4 |
| Grad clip | 1.0 (unscaled first) |
| Gradient checkpointing | on (activations of frozen layers 0–19 are needed for backprop) |
| Precision | fp16 autocast + GradScaler; fp32 norms/gates |
| Checkpointing | **trainable state dict only** (projectors + xattn + gates) + config |

**Turing/T4 constraints (Phase 0).** No bf16, so fp16 + GradScaler instead. No FlashAttention
(`attn_implementation="sdpa"`). VRAM is dominated by activations, mitigated by gradient
checkpointing, small micro-batch + accumulation, and the cached image features.

**Sanity check.** An overfit run on 40 examples reached **EM 1.0 / F1 1.0, loss 3e-4**,
confirming the loss mask and gate gradients before the full run.

**Training dynamics.** Loss fell from ~4.4 to ~0.08 over 4 epochs. Gates opened slowly and
stayed small in magnitude (|tanh| ≲ 0.03), yet the blind ablation (§4) shows the whole score
depends on them.

---

## 4. Evaluation metrics & results

Greedy decode from `[BOS] <question>`, image supplied via cross-attention, stop on EOS.
**Metrics:** normalized exact-match (lowercase, strip punctuation/articles) as primary, plus
token-level F1. Evaluated on the **8,206-example test split** with batched, left-padded
generation. The figures below are from a local single-GPU run (RTX 5060 Ti) on the exact
Kaggle-trained checkpoint. They reproduce the training-time validation score. The same eval
shards identically across 2× T4 via strided sharding.

### Headline

| Model | Exact-match | Token-F1 |
|---|---|---|
| **Full (gates on)** | **0.608** | **0.719** |
| Blind LLM (gates forced to 0) | 0.000 | 0.050 |

The blind baseline, the identical network with the image path switched off, scores **0.000
EM**. The entire 0.608 EM comes from the visual pathway; the frozen LLM alone cannot
produce the dataset's answers from the question text.

### Per-complexity exact-match

| Complexity | EM |
|---|---|
| L1, perception | 0.957 |
| L2, grounding | 0.949 |
| L3, reasoning | 0.239 |

### Per-category exact-match

| Question type | Complexity | EM |
|---|---|---|
| Detailed Verification | L2 | 1.000 |
| Existence & Sanity Check | L1 | 0.990 |
| General Health Assessment | L1 | 0.977 |
| Plant Species Identification | L1 | 0.937 |
| Visual Attribute Grounding | L2 | 0.903 |
| Causal Reasoning | L3 | 0.264 |
| Specific Disease Identification | L3 | 0.262 |
| Comprehensive Description | L3 | 0.219 |
| Counterfactual Reasoning | L3 | 0.160 |

**Reading the results.** Perception and grounding are near-ceiling: the visual features
support species ID, health checks, and attribute verification. The L3 reasoning tail is much
harder. These questions require multi-step inference and longer free-form answers, where exact
match is stringent. A small frozen adapter over a frozen 1.5B LLM also has limited reasoning
headroom. The L1-to-L3 gap shows how far the method reaches.

### Gate-magnitude analysis (`analyze_gates.py` → `gates.png`)

![Final tanh-gate magnitudes per insertion layer](output/runs/exp1/gates.png)

Final learned gates $[\tanh\alpha, \tanh\beta]$ per insertion layer:

| Layer | $\tanh\alpha$ (attention) | $\tanh\beta$ (FFN) |
|---|---|---|
| 7  | −0.007 | −0.020 |
| 10 | −0.007 | −0.015 |
| 13 | −0.020 | +0.022 |
| 16 | +0.015 | −0.027 |
| 19 | −0.022 | +0.031 |

Gates are non-zero at every layer, so all five placements contribute, but they stay small in
absolute magnitude. The **later layers (13/16/19) open wider** than the early ones. This fits
the expectation that mid-to-upper decoder layers are where text queries best retrieve
semantic image content. The score depends entirely on the gates even though they stay small.
That suggests the `tanh`-gated residual works as a small correction to an already-fluent
language model rather than a rewrite of the residual stream.

---

## 5. Bias analysis

Bias in this system comes from the dataset's construction (templated, PlantVillage-sourced)
and surfaces unevenly across question types. Three findings from the evaluation above:

- **Answer-side language prior.** **42.1 %** of answers are yes/no and answers are short
  (mean 5.17 words, median 1), so the label distribution alone carries strong signal. The
  **blind-LLM baseline** (identical network, vision path forced off) isolates this: it scores
  **0.000 EM / 0.050 F1**. The prior does not let the frozen LLM shortcut the task, so all
  accuracy is visual. But the templated answer space still means normalized exact-match
  can overstate genuine understanding on the easy, high-frequency categories.
- **Category / complexity performance disparity.** Accuracy is highly unequal across the 9
  question types. Perception and grounding (L1/L2) sit near ceiling (EM 0.90–1.00) while every
  L3 reasoning category collapses (EM 0.16–0.27; see §4 tables). A user asking a
  disease-identification or causal question receives a far less reliable answer than one asking
  a species or health-check question. That is a fairness concern if outputs are treated uniformly.
- **Source-corpus representation bias.** PlantVillage covers **14 crops / 38 diseases** of
  lab-captured single leaves on uniform backgrounds. Crops, diseases, growth stages, lighting,
  and field conditions outside that set are unrepresented. Accuracy on any real-world
  distribution that differs from the lab corpus is unknown and expected to be lower.

---

## 6. Ethical considerations

- **Over-trust / real-world harm.** A fluent but wrong answer about a diseased crop can drive
  bad crop-management or treatment decisions. If acted on, the economic and food-security cost
  is real. Fluency should not be read as reliability, so outputs must be gated by a human expert.
- **Uneven reliability across users' questions.** Because reasoning-type questions are far less
  accurate (§5), the model can appear confident while being wrong on exactly the harder
  diagnostic questions a practitioner most needs help with.
- **Attribution & licensing.** Training data is PlantVillageVQA under CC BY 4.0; downstream use
  must preserve attribution to Sakib et al. and the PlantVillage source corpus (see §10).

---

## 7. Limitations

- **Weak L3 reasoning** (EM ≈ 0.24): unreliable on causal/counterfactual and
  disease-identification questions; treat any such output as low-confidence.
- **Lab-image domain gap.** Trained and evaluated only on PlantVillage lab images (uniform
  background, single leaf); not validated on field/phone photos.
- **Narrow corpus.** Not validated outside the 14 crops / 38 diseases of the source corpus.
- **Capacity & precision limits.** A small frozen adapter over a frozen 1.5B LLM has limited
  reasoning headroom; the **small ViT** and **fp16 numeric limits** further cap accuracy.
- **Metric limits.** Normalized exact-match is stringent on longer free-form answers and
  lenient on short templated ones, so a single headline number hides the L1↔L3 spread.

---

## 8. Appropriate use cases

**Appropriate.**
- Research/education on adapter-based multimodal coupling (Flamingo-style gated cross-attention
  over frozen backbones).
- VQA over PlantVillage-style *lab* leaf images (single leaf, controlled background) for
  perception/grounding questions (species ID, health assessment, attribute verification), where
  accuracy is near-ceiling, used as an assistive, human-reviewed signal.

**Inappropriate / out of scope.**
- **Not a substitute for agronomist diagnosis** or any real crop-management/treatment decision.
- **Not for field images.** A substantial domain gap exists for phone photos in the field.
- **Not for crops/diseases outside** the 14 crops / 38 diseases of the source corpus.
- **Not for standalone L3 reasoning** (disease identification, causal/counterfactual questions)
  without expert review, given EM ≈ 0.24.

---

## 9. Reproduction

```bash
# 1. prepare a stratified 30k subset, split by image
python -m data.prepare --config configs/default.yaml --subset-size 30000
# 2. cache frozen SigLIP features (one .pt per image)
python -m data.cache_features --config configs/default.yaml
# 3. train (single GPU; fp16 + grad checkpointing + cosine/warmup)
python train.py --config configs/default.yaml            # --resume to continue a timed-out run
# 4. evaluate + blind baseline (optionally --gpus 2 to shard across both T4s)
python eval.py --config configs/default.yaml --ckpt runs/exp1/best.pt --split test
python eval.py --config configs/default.yaml --ckpt runs/exp1/best.pt --split test --blind
# 5. gate figure
python analyze_gates.py --ckpt runs/exp1/best.pt --out runs/exp1/gates.png
```

Overfit sanity: `python train.py --overfit 40 --grad-accum 1 --epochs 40` → EM should reach 1.0.

---

## 10. Citation

```bibtex
@article{sakib2025plantvillagevqa,
  title  = {PlantVillageVQA: A Visual Question Answering Dataset for Plant Disease Diagnosis},
  author = {Sakib, Syed Nazmus and others},
  journal= {arXiv preprint arXiv:2508.17117},
  year   = {2025}
}
@inproceedings{alayrac2022flamingo,
  title  = {Flamingo: a Visual Language Model for Few-Shot Learning},
  author = {Alayrac, Jean-Baptiste and others},
  booktitle = {NeurIPS},
  year   = {2022}
}
@article{mitchell2019modelcards,
  title  = {Model Cards for Model Reporting},
  author = {Mitchell, Margaret and others},
  journal= {FAT*},
  year   = {2019}
}
```

**Backbones:** MiniCPM5-1B (OpenBMB), SigLIP ViT-B/16 (`google/siglip-base-patch16-224`).
**Dataset license:** CC BY 4.0, with attribution to Sakib et al. and the PlantVillage source corpus.

---

## Appendix A. The mathematics

*(Supporting detail for §3 Training method; the architecture equations.)*

### A.1 Setup and notation

Let a decoder hidden state at an insertion layer be $H \in \mathbb{R}^{B \times T \times d}$
with $d = 1536$, $T$ the text sequence length. Let the frozen ViT produce patch tokens
$V \in \mathbb{R}^{B \times N \times d_v}$ with $N = 196$, $d_v = 768$ (the **last hidden
state**, not the pooled CLS, because cross-attention needs the spatial set).

### A.2 Per-layer projector

Each insertion layer $\ell$ owns a projector $P_\ell : \mathbb{R}^{d_v} \to \mathbb{R}^{d}$

$$
P_\ell(V) = \mathrm{LN}\big(W_2\,\phi(W_1 V + b_1) + b_2\big),\qquad
W_1 \in \mathbb{R}^{d\times d_v},\; W_2 \in \mathbb{R}^{d\times d},
$$

with $\phi = \mathrm{GELU}$. Projectors are **per-layer and identical in size** (uniform
capacity is fixed up front, not tuned per depth). This yields
$\tilde V_\ell = P_\ell(V) \in \mathbb{R}^{B\times N\times d}$.

### A.3 Gated cross-attention block

Text is the **query**, image tokens are **keys/values**. With multi-head attention at
$h = 16$ heads (head dim $d/h = 96$):

$$
A = \mathrm{MHA}\big(Q=\mathrm{LN}_q(H),\; K=\tilde V_\ell,\; V=\tilde V_\ell\big)
$$

$$
H' = H + \tanh(\alpha)\cdot A
$$

$$
H'' = H' + \tanh(\beta)\cdot \mathrm{FFN}\big(\mathrm{LN}_f(H')\big),\qquad
\mathrm{FFN}(x) = W_4\,\mathrm{GELU}(W_3 x),\; W_3\in\mathbb{R}^{4d\times d}
$$

$\alpha, \beta \in \mathbb{R}$ are scalar gates **initialised to 0**. Because
$\tanh(0)=0$, at initialisation $H'' = H$ exactly: the block is a strict identity and the
whole network reduces to the untouched frozen LLM. The $\tanh$ (bounded in $(-1,1)$) keeps
the gated residual from exploding early in training when gradients through the fresh block
are noisy.

**Design notes with mathematical justification.**
- **No RoPE on image tokens.** RoPE injects a rotation $R_\theta(m)$ that depends on a token's
  *linear position* $m$. Image patch keys form an unordered set, since their 2-D layout is
  already encoded by SigLIP's own positional embeddings. A 1-D rotation on them is meaningless.
  The text stream's RoPE is left untouched, and cross-attention keys/values carry none.
- **Fresh MHA, standard convention.** Because these are *new* blocks (not reusing MiniCPM's
  `o_proj`), there is no need to replicate the LLM's decoupled `head_dim=128` / GQA KV-head
  scheme; those are self-attention KV-cache optimisations. A plain $d=1536$, 16-head MHA is
  used.
- **fp32 gates/norms.** $\alpha,\beta$ and all LayerNorms run in fp32 for numerical stability
  under fp16 autocast.

### A.4 Insertion placement

Blocks sit at layers $\{7,10,13,16,19\}$ of 24, the **semantic middle**. Layers 0–6 handle
largely syntactic/local structure, where a text query is a poor retriever of relevant image
content. The top layers are left to integrate. Five evenly spaced blocks trade coverage
against the VRAM cost of back-propagating through frozen layers below each insertion point.

### A.5 Training objective

Sequences are laid out as

$$
[\mathrm{BOS}]\; \underbrace{q_1 \dots q_m}_{\text{question}}\; \underbrace{a_1 \dots a_k}_{\text{answer}}\; [\mathrm{EOS}]
$$

and the loss is next-token cross-entropy **masked to the answer span + EOS only**:

$$
\mathcal{L} = -\frac{1}{|\mathcal{S}|}\sum_{t\in\mathcal{S}} \log p_\theta\big(y_t \mid y_{<t}, V\big),
\qquad \mathcal{S} = \{\text{answer tokens}\}\cup\{\mathrm{EOS}\}
$$

Labels for BOS and every question token are set to $-100$ (ignored). Only
$\theta = \{\text{projectors}, \text{cross-attention}, \alpha, \beta\}$ receive gradients;
the LLM's own `lm_head` is used so any MiniCPM logit scaling is respected.
