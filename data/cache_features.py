"""Phase 1.4: precompute & cache frozen SigLIP features for all subset images.

ViT is frozen, so features are constant -> compute once, store one fp16 [196,768]
tensor per image_id. Training then skips the ViT forward entirely.

Usage:
    python -m data.cache_features --config configs/default.yaml
"""
import argparse
import json
import os

import torch
from PIL import Image

from data.prepare import load_config
from models.vision import VisionEncoder


def read_manifest(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    manifest = os.path.join(cfg["data"]["prepared_dir"], "images.jsonl")
    items = read_manifest(manifest)
    out_dir = cfg["cache"]["feature_dir"]
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = VisionEncoder(cfg["model"]["vision_name"], device=device,
                        dtype=torch.float16 if device == "cuda" else torch.float32)

    bs = cfg["cache"]["batch_size"]
    todo = [it for it in items
            if not os.path.exists(os.path.join(out_dir, _key(it["image_id"]) + ".pt"))]
    print(f"[cache] {len(items)} images, {len(todo)} to compute, device={device}")

    buf_imgs, buf_ids = [], []
    done = 0
    for it in todo:
        try:
            img = Image.open(it["disk_path"]).convert("RGB")
        except Exception as e:
            print(f"[cache] skip {it['image_id']}: {e}")
            continue
        buf_imgs.append(img)
        buf_ids.append(it["image_id"])
        if len(buf_imgs) == bs:
            _flush(enc, buf_imgs, buf_ids, out_dir)
            done += len(buf_ids)
            buf_imgs, buf_ids = [], []
            if done % (bs * 10) == 0:
                print(f"[cache] {done}/{len(todo)}")
    if buf_imgs:
        _flush(enc, buf_imgs, buf_ids, out_dir)
        done += len(buf_ids)
    print(f"[cache] done, {done} new features in {out_dir}")


def _key(image_id):
    return os.path.splitext(str(image_id))[0].lower()


def _flush(enc, imgs, ids, out_dir):
    feats = enc.encode_pil(imgs).to(torch.float16).cpu()  # [B,196,768]
    for i, image_id in enumerate(ids):
        torch.save(feats[i].clone(), os.path.join(out_dir, _key(image_id) + ".pt"))


if __name__ == "__main__":
    main()
