"""Phase 4: evaluation — generation + metrics.

Metrics: normalized exact-match (primary), token-level F1, per-category and
per-complexity accuracy. Generation is greedy (or beam) from [BOS] <question>,
image supplied via cross-attention.
"""
import argparse
import os
import re
import string
from collections import defaultdict

import torch

_ARTICLES = {"a", "an", "the"}


def normalize(s):
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    toks = [t for t in s.split() if t not in _ARTICLES]
    return " ".join(toks)


def token_f1(pred, gold):
    p = normalize(pred).split()
    g = normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = defaultdict(int)
    gc = defaultdict(int)
    for t in g:
        gc[t] += 1
    match = 0
    for t in p:
        if gc[t] > 0:
            gc[t] -= 1
            match += 1
    if match == 0:
        return 0.0
    prec = match / len(p)
    rec = match / len(g)
    return 2 * prec * rec / (prec + rec)


@torch.no_grad()
def generate_batch(model, batch, cfg, device, blind=False):
    """Greedy/beam decode answers. Left-context = [BOS] <question>; the model's
    cross-attention supplies the image. Returns list[str] predictions."""
    model.eval()
    tok = model.tokenizer
    ecfg = cfg["eval"]
    eos_id = cfg["model"]["eos_id"]
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else 0

    img = batch["image_features"].to(device, torch.float16)
    preds = []
    # Build per-example prompt = [BOS] + question tokens (no answer).
    for i in range(len(batch["questions"])):
        q_ids = tok.encode(batch["questions"][i], add_special_tokens=False)
        q_ids = q_ids[:cfg["tokenizer"]["max_question_tokens"]]
        prompt = torch.tensor([[bos_id] + q_ids], device=device)
        attn = torch.ones_like(prompt)
        model._blind = blind
        model.set_image_features(None if blind else img[i:i + 1])
        try:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                gen = model.llm.generate(
                    input_ids=prompt, attention_mask=attn,
                    max_new_tokens=ecfg["max_new_tokens"],
                    num_beams=ecfg["num_beams"],
                    do_sample=False,
                    eos_token_id=eos_id, pad_token_id=cfg["model"]["pad_id"])
        finally:
            model.clear_image_features()
            model._blind = False
        new = gen[0, prompt.shape[1]:]
        preds.append(tok.decode(new, skip_special_tokens=True).strip())
    return preds


@torch.no_grad()
def evaluate(model, loader, cfg, device, max_batches=None, blind=False):
    em_total = f1_total = n = 0
    by_cat = defaultdict(lambda: [0, 0])
    by_cplx = defaultdict(lambda: [0, 0])
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        preds = generate_batch(model, batch, cfg, device, blind=blind)
        for pred, gold, cat, cplx in zip(
                preds, batch["answers"], batch["question_types"],
                batch["complexities"]):
            em = 1.0 if normalize(pred) == normalize(gold) else 0.0
            em_total += em
            f1_total += token_f1(pred, gold)
            n += 1
            by_cat[cat][0] += em; by_cat[cat][1] += 1
            by_cplx[cplx][0] += em; by_cplx[cplx][1] += 1
    n = max(1, n)
    return {
        "exact_match": em_total / n,
        "token_f1": f1_total / n,
        "n": n,
        "per_category_em": {k: round(v[0] / v[1], 4) for k, v in by_cat.items()},
        "per_complexity_em": {k: round(v[0] / v[1], 4) for k, v in by_cplx.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--blind", action="store_true", help="blind-LLM baseline (gates off)")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--dump", default=None, help="write qualitative predictions jsonl")
    args = ap.parse_args()

    from data.prepare import load_config
    from data.dataset import VQADataset, make_collate
    from models.model import VQAModel
    from torch.utils.data import DataLoader
    import json

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VQAModel(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_trainable_state_dict(ckpt["trainable"])
    model.eval()

    ds = VQADataset(os.path.join(cfg["data"]["prepared_dir"], f"{args.split}.jsonl"),
                    model.tokenizer, cfg, cfg["cache"]["feature_dir"])
    loader = DataLoader(ds, batch_size=cfg["train"]["micro_batch"], shuffle=False,
                        collate_fn=make_collate(cfg["model"]["pad_id"]))

    metrics = evaluate(model, loader, cfg, device, max_batches=args.max_batches,
                       blind=args.blind)
    print(json.dumps(metrics, indent=2))
    gates = model.gate_report()
    print("gates (tanh alpha, beta):",
          {k: [round(a, 4), round(b, 4)] for k, (a, b) in gates.items()})


if __name__ == "__main__":
    main()
