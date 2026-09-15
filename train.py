"""Phase 3: single-GPU training (fp16 autocast + GradScaler).

Trains only the projectors + gated cross-attention blocks; the LLM and ViT are
frozen and image features are read from the precomputed cache.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --overfit 50   # sanity slice
"""
import argparse
import json
import math
import os

import torch
from torch.utils.data import DataLoader, Subset

from data.prepare import load_config
from data.dataset import VQADataset, make_collate
from models.model import VQAModel
from eval import evaluate


def build_loader(cfg, split, tokenizer, shuffle):
    ds = VQADataset(
        os.path.join(cfg["data"]["prepared_dir"], f"{split}.jsonl"),
        tokenizer, cfg, cfg["cache"]["feature_dir"])
    loader = DataLoader(
        ds, batch_size=cfg["train"]["micro_batch"], shuffle=shuffle,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
        collate_fn=make_collate(cfg["model"]["pad_id"]))
    return ds, loader


def param_groups(model, weight_decay):
    """No weight decay on gates (alpha/beta) and LayerNorms/biases."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith("alpha") or n.endswith("beta") or "ln" in n.lower() \
                or "norm" in n.lower() or n.endswith("bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--overfit", type=int, default=0,
                    help="if >0, train+eval on this many train examples (sanity)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["train"]
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.grad_accum is not None:
        tcfg["grad_accum"] = args.grad_accum
    if args.lr is not None:
        tcfg["lr"] = args.lr
    torch.manual_seed(tcfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "training expects CUDA"

    model = VQAModel(cfg).to(device)
    if tcfg["gradient_checkpointing"]:
        model.llm.gradient_checkpointing_enable()
        model.llm.config.use_cache = False

    train_ds, train_loader = build_loader(cfg, "train", model.tokenizer, shuffle=True)
    if args.overfit:
        train_ds = Subset(train_ds, list(range(args.overfit)))
        train_loader = DataLoader(
            train_ds, batch_size=tcfg["micro_batch"], shuffle=True,
            collate_fn=make_collate(cfg["model"]["pad_id"]))
        val_loader = train_loader  # overfit sanity: eval on the same slice
    else:
        _, val_loader = build_loader(cfg, "val", model.tokenizer, shuffle=False)

    opt = torch.optim.AdamW(param_groups(model, tcfg["weight_decay"]), lr=tcfg["lr"])
    scaler = torch.cuda.amp.GradScaler()

    steps_per_epoch = math.ceil(len(train_loader) / tcfg["grad_accum"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    warmup = max(1, int(total_steps * tcfg["warmup_frac"]))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.makedirs(tcfg["out_dir"], exist_ok=True)
    log_path = os.path.join(tcfg["out_dir"], "train_log.jsonl")
    logf = open(log_path, "a", encoding="utf-8")

    best_em = -1.0
    gstep = 0
    eval_every = max(1, int(steps_per_epoch * cfg["train"]["eval_every_frac"]))

    for epoch in range(tcfg["epochs"]):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for it, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            img = batch["image_features"].to(device, torch.float16)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(input_ids, attn, img, labels=labels)
                loss = out.loss / tcfg["grad_accum"]
            scaler.scale(loss).backward()
            running += out.loss.item()

            if (it + 1) % tcfg["grad_accum"] == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    tcfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                gstep += 1

                if gstep % cfg["train"]["log_every"] == 0:
                    avg = running / (tcfg["grad_accum"] * cfg["train"]["log_every"])
                    running = 0.0
                    gates = model.gate_report()
                    rec = {"epoch": epoch, "step": gstep,
                           "loss": round(avg, 4),
                           "lr": round(sched.get_last_lr()[0], 6),
                           "gates_alpha": {k: round(v[0], 4) for k, v in gates.items()}}
                    print(f"[train] ep{epoch} step{gstep} loss={avg:.4f} "
                          f"lr={rec['lr']} alpha={rec['gates_alpha']}")
                    logf.write(json.dumps(rec) + "\n"); logf.flush()

                if gstep % eval_every == 0:
                    em = run_eval_and_save(model, val_loader, cfg, device,
                                           gstep, logf, best_em)
                    best_em = max(best_em, em)
                    model.train()

        em = run_eval_and_save(model, val_loader, cfg, device, gstep, logf, best_em)
        best_em = max(best_em, em)
        model.train()

    logf.close()
    print(f"[train] done. best val EM={best_em:.4f}")


def run_eval_and_save(model, loader, cfg, device, gstep, logf, best_em):
    metrics = evaluate(model, loader, cfg, device, max_batches=None)
    gates = model.gate_report()
    rec = {"step": gstep, "eval": metrics,
           "gates": {k: [round(a, 4), round(b, 4)] for k, (a, b) in gates.items()}}
    logf.write(json.dumps(rec) + "\n"); logf.flush()
    print(f"[eval] step{gstep} EM={metrics['exact_match']:.4f} "
          f"F1={metrics['token_f1']:.4f} gates={rec['gates']}")

    ckpt_dir = cfg["train"]["out_dir"]
    torch.save({"trainable": model.trainable_state_dict(),
                "cfg": cfg, "step": gstep, "metrics": metrics},
               os.path.join(ckpt_dir, "last.pt"))
    if metrics["exact_match"] > best_em:
        torch.save({"trainable": model.trainable_state_dict(),
                    "cfg": cfg, "step": gstep, "metrics": metrics},
                   os.path.join(ckpt_dir, "best.pt"))
        print(f"[eval] new best EM {metrics['exact_match']:.4f} -> best.pt")
    return metrics["exact_match"]


if __name__ == "__main__":
    main()
