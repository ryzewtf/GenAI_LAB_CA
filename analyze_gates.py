"""Phase 4: gate-analysis figure — bar chart of final tanh(alpha) per insertion layer.

Reads a checkpoint's config + the model's gate report (or the train_log.jsonl last
eval entry) and writes a PNG for the model card.

Usage:
    python analyze_gates.py --ckpt runs/exp1/best.pt --out runs/exp1/gates.png
    python analyze_gates.py --log runs/exp1/train_log.jsonl --out runs/exp1/gates.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def from_log(log_path):
    last = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if "gates" in rec:
                last = rec["gates"]
    return {int(k): v for k, v in last.items()} if last else None


def from_ckpt(ckpt_path):
    import torch
    from models.model import VQAModel
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    model = VQAModel(cfg)
    model.load_trainable_state_dict(ckpt["trainable"])
    rep = model.gate_report()
    return {li: [a, b] for li, (a, b) in rep.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", default="gates.png")
    args = ap.parse_args()

    gates = from_ckpt(args.ckpt) if args.ckpt else from_log(args.log)
    assert gates, "no gate data found"
    layers = sorted(gates)
    alpha = [abs(gates[l][0]) for l in layers]
    beta = [abs(gates[l][1]) for l in layers]

    x = range(len(layers))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - w / 2 for i in x], alpha, w, label="|tanh(alpha)| (attention gate)")
    ax.bar([i + w / 2 for i in x], beta, w, label="|tanh(beta)| (ffn gate)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_xlabel("Insertion layer")
    ax.set_ylabel("Gate magnitude")
    ax.set_title("Learned gated cross-attention magnitudes per layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[gates] wrote {args.out}")
    for l in layers:
        print(f"  L{l}: alpha={gates[l][0]:+.4f}  beta={gates[l][1]:+.4f}")


if __name__ == "__main__":
    main()
