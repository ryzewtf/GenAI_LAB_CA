"""Phase 1.1-1.2: load, inspect, subset, split-by-image.

Reads the PlantVillageVQA CSV (Kaggle-mounted or locally extracted), joins each QA
pair to an image file on disk (by image_id, case-insensitively), derives a 3-level
`complexity` from the 9 question_type categories, builds a stratified subset, and
splits BY IMAGE (never by QA pair) into train/val/test manifests (JSONL).

Usage:
    python -m data.prepare --config configs/default.yaml [--subset-size N]
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import pandas as pd
import yaml


# 9 question_type categories -> 3 cognitive levels (README taxonomy).
COMPLEXITY_MAP = {
    "Existence & Sanity Check": "L1_perception",
    "Plant Species Identification": "L1_perception",
    "General Health Assessment": "L1_perception",
    "Visual Attribute Grounding": "L2_grounding",
    "Detailed Verification": "L2_grounding",
    "Specific Disease Identification": "L3_reasoning",
    "Comprehensive Description": "L3_reasoning",
    "Causal Reasoning": "L3_reasoning",
    "Counterfactual Reasoning": "L3_reasoning",
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def index_images(images_dir):
    """Map lowercased basename-without-ext -> actual relative path on disk.

    The zip mixes .JPG/.jpg case and the CSV's image_path does not match the disk
    layout, so we join on image_id stem instead.
    """
    index = {}
    for root, _, files in os.walk(images_dir):
        for fn in files:
            stem = os.path.splitext(fn)[0].lower()
            index[stem] = os.path.join(root, fn)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--subset-size", type=int, default=None,
                    help="override data.subset_size")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    if args.subset_size is not None:
        dcfg["subset_size"] = args.subset_size
    random.seed(dcfg["seed"])

    data_root = dcfg["data_root"]
    csv_path = os.path.join(data_root, dcfg["csv_name"])
    images_dir = os.path.join(data_root, dcfg["images_dirname"])
    print(f"[prepare] csv={csv_path}")
    print(f"[prepare] images_dir={images_dir}")

    df = pd.read_csv(csv_path)
    print(f"[prepare] loaded {len(df)} QA rows; columns={list(df.columns)}")

    # --- join to images on disk by image_id stem (case-insensitive) ---
    img_index = index_images(images_dir)
    print(f"[prepare] found {len(img_index)} image files on disk")

    def resolve(image_id):
        stem = os.path.splitext(str(image_id))[0].lower()
        return img_index.get(stem)

    df["disk_path"] = df["image_id"].map(resolve)
    missing = df["disk_path"].isna().sum()
    if missing:
        print(f"[prepare] WARNING: {missing} QA rows have no image file; dropping them")
        df = df[df["disk_path"].notna()].reset_index(drop=True)

    df["complexity"] = df["question_type"].map(COMPLEXITY_MAP)
    unmapped = df["complexity"].isna().sum()
    if unmapped:
        raise ValueError(f"{unmapped} rows have an unmapped question_type: "
                         f"{sorted(set(df.loc[df['complexity'].isna(), 'question_type']))}")

    # --- inspection log (feeds the model card) ---
    print("\n=== INSPECTION ===")
    print(f"QA pairs: {len(df)} | unique images: {df['image_id'].nunique()}")
    print("question_type counts:")
    for k, v in Counter(df["question_type"]).most_common():
        print(f"  {k:32s} {v}")
    print("complexity counts:", dict(Counter(df["complexity"])))
    print("native split counts:", dict(Counter(df["split"])))
    alens = df["answer"].astype(str).str.split().map(len)
    print(f"answer word-len min/mean/max: {alens.min()}/{alens.mean():.1f}/{alens.max()}")

    # --- stratified subset by question_type, capping unique images ---
    subset_size = dcfg["subset_size"]
    max_images = dcfg["max_images"]

    # First cap images: keep a random subset of unique images so the feature cache
    # stays small. Sampling QA within kept images preserves image-level integrity.
    all_images = df["image_id"].unique().tolist()
    random.shuffle(all_images)
    kept_images = set(all_images[:max_images])
    df = df[df["image_id"].isin(kept_images)].reset_index(drop=True)
    print(f"\n[prepare] capped to {len(kept_images)} images -> {len(df)} QA rows")

    if subset_size < len(df):
        # stratified by question_type, proportional allocation
        frac = subset_size / len(df)
        parts = []
        for qt, g in df.groupby("question_type"):
            n = max(1, round(len(g) * frac))
            parts.append(g.sample(n=min(n, len(g)), random_state=dcfg["seed"]))
        sub = pd.concat(parts).reset_index(drop=True)
        print(f"[prepare] stratified subset: {len(sub)} QA rows (target {subset_size})")
    else:
        sub = df
        print(f"[prepare] subset_size >= available; using all {len(sub)} rows")

    # --- split BY IMAGE ---
    # Keep the dataset's native 'test' rows as part of test; carve val+test from the
    # native 'train' images so no image crosses splits.
    imgs_by_split = defaultdict(set)
    for _, r in sub.iterrows():
        imgs_by_split[r["split"]].add(r["image_id"])

    train_imgs = list(imgs_by_split.get("train", set()))
    random.shuffle(train_imgs)
    n_val = int(len(train_imgs) * dcfg["val_frac"])
    n_test = int(len(train_imgs) * dcfg["test_frac"])
    val_imgs = set(train_imgs[:n_val])
    test_imgs = set(train_imgs[n_val:n_val + n_test])
    tr_imgs = set(train_imgs[n_val + n_test:])
    # native test images all go to test
    test_imgs |= imgs_by_split.get("test", set())

    def assign(image_id):
        if image_id in tr_imgs:
            return "train"
        if image_id in val_imgs:
            return "val"
        return "test"

    sub["final_split"] = sub["image_id"].map(assign)

    # sanity: no image in two splits
    overlap = set()
    seen = {}
    for _, r in sub.iterrows():
        s = seen.get(r["image_id"])
        if s is not None and s != r["final_split"]:
            overlap.add(r["image_id"])
        seen[r["image_id"]] = r["final_split"]
    assert not overlap, f"image leakage across splits: {list(overlap)[:5]}"

    out_dir = dcfg["prepared_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cols = ["image_id", "disk_path", "question_type", "complexity",
            "question", "answer"]
    for split in ["train", "val", "test"]:
        part = sub[sub["final_split"] == split]
        path = os.path.join(out_dir, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for _, r in part.iterrows():
                f.write(json.dumps({c: r[c] for c in cols}, ensure_ascii=False) + "\n")
        print(f"[prepare] wrote {len(part):6d} QA / {part['image_id'].nunique():5d} imgs -> {path}")

    # also dump the unique-image manifest for feature caching
    man = os.path.join(out_dir, "images.jsonl")
    with open(man, "w", encoding="utf-8") as f:
        for image_id, dp in sub[["image_id", "disk_path"]].drop_duplicates().values:
            f.write(json.dumps({"image_id": image_id, "disk_path": dp}) + "\n")
    print(f"[prepare] wrote image manifest ({sub['image_id'].nunique()} images) -> {man}")


if __name__ == "__main__":
    main()
