"""Phase 4: evaluation — generation + metrics.

Metrics: normalized exact-match (primary), token-level F1, per-category and
per-complexity accuracy. Generation is greedy (or beam) from [BOS] <question>,
image supplied via cross-attention.

Speed:
  * Batched generation — the cross-attention consumes image features row-aligned
    with the token batch, so prompts are left-padded and decoded in one call
    instead of one example at a time.
  * Optional 2-GPU sharding (`--gpus 2`) — the split is divided into contiguous
    shards, one worker process per GPU, and the raw counts are merged.
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
    """Greedy/beam decode answers for the whole batch at once.

    Prompts are `[BOS] <question>`, left-padded so the batch aligns; the model's
    cross-attention supplies each row's image. Returns list[str] predictions.
    """
    model.eval()
    tok = model.tokenizer
    ecfg = cfg["eval"]
    eos_id = cfg["model"]["eos_id"]
    pad_id = cfg["model"]["pad_id"]
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else 0
    max_q = cfg["tokenizer"]["max_question_tokens"]

    questions = batch["questions"]
    # Tokenize prompts and left-pad to the batch max (generation needs left pad).
    seqs = []
    for q in questions:
        q_ids = tok.encode(q, add_special_tokens=False)[:max_q]
        seqs.append([bos_id] + q_ids)
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, maxlen - len(s):] = torch.tensor(s, dtype=torch.long)
        attn[i, maxlen - len(s):] = 1
    input_ids = input_ids.to(device)
    attn = attn.to(device)

    img = batch["image_features"].to(device, torch.float16)
    model._blind = blind
    model.set_image_features(None if blind else img)
    try:
        with torch.cuda.amp.autocast(dtype=torch.float16):
            gen = model.llm.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=ecfg["max_new_tokens"],
                num_beams=ecfg["num_beams"],
                do_sample=False,
                eos_token_id=eos_id, pad_token_id=pad_id)
    finally:
        model.clear_image_features()
        model._blind = False

    new = gen[:, input_ids.shape[1]:]
    return [tok.decode(row, skip_special_tokens=True).strip() for row in new]


@torch.no_grad()
def evaluate_counts(model, loader, cfg, device, max_batches=None, blind=False):
    """Raw (unaveraged) counts, so shards from multiple GPUs can be merged."""
    em_total = f1_total = 0.0
    n = 0
    by_cat = defaultdict(lambda: [0.0, 0])
    by_cplx = defaultdict(lambda: [0.0, 0])
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
    return {"em": em_total, "f1": f1_total, "n": n,
            "by_cat": {k: list(v) for k, v in by_cat.items()},
            "by_cplx": {k: list(v) for k, v in by_cplx.items()}}


def finalize(counts):
    n = max(1, counts["n"])
    return {
        "exact_match": counts["em"] / n,
        "token_f1": counts["f1"] / n,
        "n": counts["n"],
        "per_category_em": {k: round(v[0] / v[1], 4)
                            for k, v in counts["by_cat"].items()},
        "per_complexity_em": {k: round(v[0] / v[1], 4)
                              for k, v in counts["by_cplx"].items()},
    }


def merge_counts(parts):
    out = {"em": 0.0, "f1": 0.0, "n": 0,
           "by_cat": defaultdict(lambda: [0.0, 0]),
           "by_cplx": defaultdict(lambda: [0.0, 0])}
    for c in parts:
        out["em"] += c["em"]; out["f1"] += c["f1"]; out["n"] += c["n"]
        for k, v in c["by_cat"].items():
            out["by_cat"][k][0] += v[0]; out["by_cat"][k][1] += v[1]
        for k, v in c["by_cplx"].items():
            out["by_cplx"][k][0] += v[0]; out["by_cplx"][k][1] += v[1]
    out["by_cat"] = {k: list(v) for k, v in out["by_cat"].items()}
    out["by_cplx"] = {k: list(v) for k, v in out["by_cplx"].items()}
    return out


@torch.no_grad()
def evaluate(model, loader, cfg, device, max_batches=None, blind=False):
    """Single-process metrics dict (used by train.py's in-loop eval)."""
    return finalize(evaluate_counts(model, loader, cfg, device,
                                    max_batches=max_batches, blind=blind))


# ---------------------------------------------------------------------------
# Multi-GPU sharded evaluation
# ---------------------------------------------------------------------------

def _build_model_and_loader(cfg, ckpt_path, split, device, batch_size,
                            indices=None, sort_by_len=True):
    from data.dataset import VQADataset, make_collate
    from models.model import VQAModel
    from torch.utils.data import DataLoader, Subset

    model = VQAModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_trainable_state_dict(ckpt["trainable"])
    model.eval()
    model.llm.config.use_cache = True  # KV cache on for generation speed

    ds = VQADataset(os.path.join(cfg["data"]["prepared_dir"], f"{split}.jsonl"),
                    model.tokenizer, cfg, cfg["cache"]["feature_dir"])
    # Length-bucketed order: group similar-length questions into the same batch
    # so left-padding is minimal and greedy decoding isn't dragged out by one
    # long row. Metrics are per-example, so reordering does not change results.
    if sort_by_len:
        base = indices if indices is not None else range(len(ds))
        order = sorted(base, key=lambda i: len(str(ds.rows[i]["question"]).split()))
        ds = Subset(ds, order)
    elif indices is not None:
        ds = Subset(ds, indices)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=cfg["train"].get("num_workers", 2),
                        pin_memory=True,
                        collate_fn=make_collate(cfg["model"]["pad_id"]))
    return model, loader


def _worker(rank, world, cfg, ckpt_path, split, blind, batch_size,
            max_batches, n_total, q):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    # strided shard so the two GPUs get balanced, interleaved work
    shard = list(range(rank, n_total, world))
    model, loader = _build_model_and_loader(
        cfg, ckpt_path, split, device, batch_size, indices=shard)
    counts = evaluate_counts(model, loader, cfg, device,
                             max_batches=max_batches, blind=blind)
    q.put(counts)


def evaluate_multigpu(cfg, ckpt_path, split, gpus, blind, batch_size,
                      max_batches=None):
    import torch.multiprocessing as mp
    from data.dataset import VQADataset

    # length of the split (build the dataset once, cheaply, on CPU tokenizer)
    from models.model import VQAModel  # noqa: F401  (import cost amortized in workers)
    # count lines instead of constructing the full dataset/model here
    jl = os.path.join(cfg["data"]["prepared_dir"], f"{split}.jsonl")
    with open(jl, "r", encoding="utf-8") as f:
        n_total = sum(1 for _ in f)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for rank in range(gpus):
        p = ctx.Process(target=_worker, args=(
            rank, gpus, cfg, ckpt_path, split, blind, batch_size,
            max_batches, n_total, q))
        p.start()
        procs.append(p)
    parts = [q.get() for _ in range(gpus)]
    for p in procs:
        p.join()
    return finalize(merge_counts(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--blind", action="store_true", help="blind-LLM baseline (gates off)")
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="eval generation batch size (default: eval.batch_size or 16)")
    ap.add_argument("--gpus", type=int, default=1,
                    help="number of GPUs to shard evaluation across")
    ap.add_argument("--dump", default=None, help="write qualitative predictions jsonl")
    ap.add_argument("--out", default=None,
                    help="write the final metrics dict to this JSON path")
    args = ap.parse_args()

    from data.prepare import load_config
    import json

    cfg = load_config(args.config)
    batch_size = args.batch_size or cfg.get("eval", {}).get("batch_size", 16)

    def _save(metrics):
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({"split": args.split, "blind": args.blind, **metrics},
                          f, indent=2)
            print(f"[eval] wrote metrics -> {args.out}")

    if args.gpus > 1:
        assert torch.cuda.device_count() >= args.gpus, (
            f"requested {args.gpus} GPUs, only {torch.cuda.device_count()} visible")
        metrics = evaluate_multigpu(cfg, args.ckpt, args.split, args.gpus,
                                    args.blind, batch_size,
                                    max_batches=args.max_batches)
        print(json.dumps(metrics, indent=2))
        _save(metrics)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, loader = _build_model_and_loader(
        cfg, args.ckpt, args.split, device, batch_size)
    metrics = evaluate(model, loader, cfg, device, max_batches=args.max_batches,
                       blind=args.blind)
    print(json.dumps(metrics, indent=2))
    _save(metrics)
    gates = model.gate_report()
    print("gates (tanh alpha, beta):",
          {k: [round(a, 4), round(b, 4)] for k, (a, b) in gates.items()})


if __name__ == "__main__":
    main()
