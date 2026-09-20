#!/usr/bin/env python3
"""Upload the pipeline's non-git assets to Kaggle as a single Dataset.

git carries only code; the frozen backbones, the trained checkpoint, the cached
SigLIP features and the prepared manifests must be attached to the Kaggle
notebook separately. This script bundles them into ONE Kaggle Dataset so the
notebook can mount them at a predictable path.

Auth (any one of):
  * env vars  KAGGLE_USERNAME  and  KAGGLE_KEY
  * ~/.kaggle/kaggle.json  (or a folder pointed at by KAGGLE_CONFIG_DIR)

The dataset mounts on Kaggle at:  /kaggle/input/<slug>/<same layout as below>

Layout uploaded (relative to the mount root):
  models/MiniCPM5-1B/      frozen LLM (safetensors + tokenizer + config)
  models/siglip/           frozen SigLIP (only needed to RE-cache features)
  runs/exp1/best.pt        trained adapters (projectors + xattn + gates)
  prepared/                train/val/test/images .jsonl manifests
  cache/siglip_features/   precomputed [196,768] fp16 features, one .pt/image

Usage:
  .linux-venv/bin/python upload_to_kaggle.py                 # upload all
  .linux-venv/bin/python upload_to_kaggle.py --slug agrivqa-assets
  .linux-venv/bin/python upload_to_kaggle.py --skip siglip   # leave a part out
  .linux-venv/bin/python upload_to_kaggle.py --dry-run       # stage only, no upload
"""
import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# component -> (source path, destination path under the dataset root)
# A source that is a directory is copied/linked recursively; a file is linked as-is.
COMPONENTS = {
    "llm":      (os.path.join(HERE, "output/models/MiniCPM5-1B"), "models/MiniCPM5-1B"),
    "siglip":   (os.path.join(HERE, "output/models/siglip"),      "models/siglip"),
    "ckpt":     (os.path.join(HERE, "output/runs/exp1/best.pt"),  "runs/exp1/best.pt"),
    "prepared": (os.path.join(HERE, "output/prepared"),           "prepared"),
    "features": (os.path.join(HERE, "data/cache/siglip_features"),"cache/siglip_features"),
}

# Files/dirs inside a source we never want to ship.
SKIP_NAMES = {".cache", ".huggingface", "__pycache__", ".git"}
# For siglip, the .bin is a duplicate of model.safetensors; drop it to save ~0.8GB.
SKIP_FILES = {"pytorch_model.bin"}


def _link_or_copy(src, dst):
    """Hardlink src->dst when on the same filesystem (instant, no extra disk);
    fall back to a real copy otherwise."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _stage_path(src, dst_root, rel):
    dst = os.path.join(dst_root, rel)
    if os.path.isfile(src):
        _link_or_copy(src, dst)
        return 1, os.path.getsize(src)
    n = size = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]
        for f in files:
            if f in SKIP_FILES:
                continue
            s = os.path.join(root, f)
            r = os.path.relpath(s, src)
            _link_or_copy(s, os.path.join(dst, r))
            n += 1
            size += os.path.getsize(s)
    return n, size


def _build_zip(stage_dir, zip_path):
    """Zip the staged tree with ZIP_STORED (no compression).

    The assets are already dense (.pt / .safetensors), so compression wastes CPU
    for almost no size gain; STORED is far faster. The archive is written to repo
    root and reused on retries, so a failed upload does not re-zip 6.8GB.
    """
    import zipfile
    tmp = zip_path + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    n = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED,
                         allowZip64=True) as zf:
        for root, _dirs, files in os.walk(stage_dir):
            for f in files:
                s = os.path.join(root, f)
                zf.write(s, os.path.relpath(s, stage_dir))
                n += 1
    os.replace(tmp, zip_path)  # atomic: a complete zip or nothing
    return n


def _human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="agrivqa-assets",
                    help="dataset slug (mount becomes /kaggle/input/<slug>/)")
    ap.add_argument("--stage-dir", default=os.path.join(HERE, ".kaggle_upload_stage"),
                    help="where the hardlinked upload tree is assembled")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=list(COMPONENTS), help="components to leave out")
    ap.add_argument("--user", default=None,
                    help="Kaggle username (else KAGGLE_USERNAME / kaggle.json)")
    ap.add_argument("--key", default=None,
                    help="Kaggle API key (else KAGGLE_KEY / KAGGLE_API_TOKEN / kaggle.json)")
    ap.add_argument("--zip", dest="zip_path", default=os.path.join(HERE, "agrivqa-assets.zip"),
                    help="persisted archive at repo root; reused across retries")
    ap.add_argument("--rezip", action="store_true",
                    help="rebuild the archive even if it already exists")
    ap.add_argument("--version-notes", default="assets for AgriVQA eval/train")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage + build the zip but do not upload")
    args = ap.parse_args()

    import kagglehub

    # ---- normalize credentials so kagglehub always finds them ----
    # kagglehub only reads KAGGLE_USERNAME + KAGGLE_KEY (or ~/.kaggle/kaggle.json).
    # It does NOT read KAGGLE_API_TOKEN, so we gather from every common source,
    # then WRITE a proper kaggle.json + set the env vars kagglehub expects.
    cfg_dir = os.getenv("KAGGLE_CONFIG_DIR", os.path.join(os.path.expanduser("~"), ".kaggle"))
    json_path = os.path.join(cfg_dir, "kaggle.json")

    def _clean(v):
        return v.strip() if isinstance(v, str) else v

    user = _clean(args.user) or _clean(os.getenv("KAGGLE_USERNAME"))
    key = (_clean(args.key) or _clean(os.getenv("KAGGLE_KEY"))
           or _clean(os.getenv("KAGGLE_API_TOKEN")))
    if (not user or not key) and os.path.isfile(json_path):
        import json as _json
        with open(json_path) as _f:
            j = _json.load(_f)
        user = user or _clean(j.get("username"))
        key = key or _clean(j.get("key"))

    if not args.dry_run:
        if not user or not key:
            sys.exit(
                "Missing Kaggle credentials (need BOTH username and key).\n"
                "  Kaggle -> Settings -> API -> 'Create New Token' downloads kaggle.json.\n"
                "  Then either:\n"
                "    export KAGGLE_USERNAME=<you>  KAGGLE_KEY=<key-from-kaggle.json>\n"
                "  or pass:  --user <you> --key <key>\n"
                "  or place that kaggle.json at ~/.kaggle/kaggle.json\n"
                "  NOTE: KAGGLE_API_TOKEN alone does NOT work -- kagglehub ignores it, "
                "and the key needs the username too.")
        # Make kagglehub see them regardless of how they were supplied.
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"] = key
        os.makedirs(cfg_dir, exist_ok=True)
        import json as _json
        with open(json_path, "w") as _f:
            _json.dump({"username": user, "key": key}, _f)
        os.chmod(json_path, 0o600)
        print(f"[auth] using Kaggle user '{user}' (wrote {json_path})")

    # ---- assemble the staging tree ----
    if os.path.exists(args.stage_dir):
        shutil.rmtree(args.stage_dir)
    os.makedirs(args.stage_dir)

    total_n = total_sz = 0
    print(f"[stage] building upload tree at {args.stage_dir}")
    for name, (src, rel) in COMPONENTS.items():
        if name in args.skip:
            print(f"  - {name:9s} SKIPPED")
            continue
        if not os.path.exists(src):
            print(f"  ! {name:9s} MISSING at {src} -- skipping")
            continue
        n, sz = _stage_path(src, args.stage_dir, rel)
        total_n += n
        total_sz += sz
        print(f"  + {name:9s} {n:>6d} files  {_human(sz):>9s}  -> {rel}")
    print(f"[stage] total {total_n} files, {_human(total_sz)}")

    # ---- build (or reuse) the persisted archive at repo root ----
    if os.path.isfile(args.zip_path) and not args.rezip:
        print(f"[zip] reusing existing {args.zip_path} "
              f"({_human(os.path.getsize(args.zip_path))}) -- pass --rezip to rebuild")
    else:
        print(f"[zip] building {args.zip_path} (ZIP_STORED, this writes ~{_human(total_sz)}) ...")
        zn = _build_zip(args.stage_dir, args.zip_path)
        print(f"[zip] wrote {zn} files -> {args.zip_path} "
              f"({_human(os.path.getsize(args.zip_path))})")

    if args.dry_run:
        print("[dry-run] archive ready; not uploading. Re-run without --dry-run.")
        return

    # Upload the single persisted zip: with one file kagglehub uploads it as-is
    # (no re-zip), and Kaggle extracts the archive into the dataset's file tree.
    zipdir = os.path.join(HERE, ".kaggle_upload_zipdir")
    if os.path.exists(zipdir):
        shutil.rmtree(zipdir)
    os.makedirs(zipdir)
    _link_or_copy(args.zip_path, os.path.join(zipdir, os.path.basename(args.zip_path)))

    handle = f"{user}/{args.slug}"
    print(f"[upload] {handle}  (uploading the persisted archive)")
    try:
        kagglehub.dataset_upload(handle, zipdir, version_notes=args.version_notes)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            sys.exit(
                f"\n[auth] Kaggle rejected the credentials (401). The archive is kept at\n"
                f"       {args.zip_path} so the retry will NOT re-zip.\n"
                f"       Regenerate the token (Kaggle -> Settings -> API -> Create New Token),\n"
                f"       re-export KAGGLE_KEY (or pass --key), then run this script again.")
        raise

    print("\n[done] dataset uploaded.")
    print(f"  handle : {handle}")
    print(f"  page   : https://www.kaggle.com/datasets/{handle}")
    print(f"  mount  : /kaggle/input/{args.slug}/  (attach it as a Notebook input)")
    print("  paths the notebook expects under that mount:")
    print(f"    LLM      : /kaggle/input/{args.slug}/models/MiniCPM5-1B")
    print(f"    SigLIP   : /kaggle/input/{args.slug}/models/siglip")
    print(f"    best.pt  : /kaggle/input/{args.slug}/runs/exp1/best.pt")
    print(f"    prepared : /kaggle/input/{args.slug}/prepared")
    print(f"    features : /kaggle/input/{args.slug}/cache/siglip_features")


if __name__ == "__main__":
    main()
