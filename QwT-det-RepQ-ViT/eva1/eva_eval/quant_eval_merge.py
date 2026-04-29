"""Merge shard checkpoints + run COCOEvaluator.

Reads ``<tag>/shard_*/checkpoint.pt`` (each produced by
``quant_eval_shard.py``), concatenates their per-image COCO-format
predictions into a fresh ``COCOEvaluator``, and runs ``evaluate()`` once to
write::

    <tag>/metrics.json                  — AP / per-task results
    <tag>/coco_instances_results.json   — COCO format detection list
    <tag>/instances_predictions.pth     — torch-pickled predictions
    <tag>/vis/                          — GT vs pred panels (first --n-vis)

The shard checkpoint files are kept in place; pass ``--rm-shards`` to
delete them once metrics.json is written.

Usage::

    python quant_eval_merge.py --tag w6a6_rp [--rm-shards]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

os.environ.setdefault("DETECTRON2_DATASETS",
                      "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVA_DET = os.path.join(_THIS_DIR, "..", "eva_det")
sys.path.insert(0, EVA_DET)

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--n-vis", type=int, default=20)
p.add_argument("--rm-shards", action="store_true",
               help="Delete <tag>/shard_*/checkpoint.pt after writing metrics.json.")
args = p.parse_args()

OUT = Path(__file__).parent / "results" / args.tag
assert OUT.exists(), f"no such dir {OUT}"

from detectron2.config import LazyConfig
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator

cfg = LazyConfig.load(
    f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py")
orig_name = cfg.dataloader.test.dataset.names
all_items = DatasetCatalog.get(orig_name)

# --- Collect shard checkpoints ---
shard_dirs = sorted(d for d in OUT.iterdir() if d.is_dir() and d.name.startswith("shard_"))
shard_files = []
for d in shard_dirs:
    cp = d / "checkpoint.pt"
    if cp.exists():
        shard_files.append(cp)
if not shard_files:
    sys.exit(f"no shard_*/checkpoint.pt found under {OUT}")

print(f"[merge] loading {len(shard_files)} shard checkpoints")
all_predictions = []
seen_ids: set[int] = set()
configs = []
for f in shard_files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    cfg_sig = d.get("config", {})
    configs.append((f, cfg_sig))
    preds = d.get("predictions", [])
    print(f"  {f.parent.name}: {len(preds)} predictions")
    for pr in preds:
        iid = int(pr["image_id"])
        if iid in seen_ids:
            continue   # duplicate (e.g. shard slices overlapped)
        seen_ids.add(iid)
        all_predictions.append(pr)

# Sanity: shard configs should agree on everything except shard_id.
def _stripped(c):
    c = dict(c); c.pop("shard_id", None); return c

base = _stripped(configs[0][1])
mismatched = [(f.parent.name, _stripped(c)) for f, c in configs[1:]
              if _stripped(c) != base]
if mismatched:
    print("[merge] WARNING: config mismatch across shards (expected to differ "
          "only on shard_id):")
    for name, c in mismatched:
        print(f"    {name}: {c}")

print(f"[merge] {len(all_predictions)} predictions, {len(seen_ids)} unique image_ids")

# --- Build evaluator + restrict its COCO API to images we predicted on ---
evaluator = COCOEvaluator(orig_name, tasks=("bbox", "segm"), distributed=False,
                          output_dir=str(OUT), max_dets_per_image=None)
coco = evaluator._coco_api
coco.imgs      = {k: v for k, v in coco.imgs.items() if k in seen_ids}
coco.anns      = {k: v for k, v in coco.anns.items() if v["image_id"] in seen_ids}
coco.imgToAnns = {k: v for k, v in coco.imgToAnns.items() if k in seen_ids}
coco.catToImgs = {c: [i for i in imgs if i in seen_ids]
                  for c, imgs in coco.catToImgs.items()}

evaluator.reset()
evaluator._predictions = list(all_predictions)
results = evaluator.evaluate()

print("[merge] Results:")
for task, m in results.items():
    print(f"  [{task}] AP={m.get('AP',0):.2f}  "
          f"AP50={m.get('AP50',0):.2f}  AP75={m.get('AP75',0):.2f}")

meta = base   # shared shard config (sans shard_id)
metrics_out = {"n_eval": len(seen_ids),
               "n_shards": len(shard_files),
               **meta, **results}
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"[merge] metrics.json -> {OUT/'metrics.json'}")

# --- FP comparison if available ---
fp_file = Path(__file__).parent / "results" / "fp100" / "metrics.json"
if fp_file.exists():
    with open(fp_file) as f:
        fp = json.load(f)
    print("[merge] Δ vs FP baseline:")
    for task in ("bbox", "segm"):
        if task in fp and task in results:
            d_ap   = results[task]["AP"]   - fp[task]["AP"]
            d_ap50 = results[task]["AP50"] - fp[task]["AP50"]
            print(f"    [{task}] ΔAP={d_ap:+.2f}  ΔAP50={d_ap50:+.2f}")

# --- Visualisations (OpenCV) ---
(OUT / "vis").mkdir(exist_ok=True)
metadata = MetadataCatalog.get(orig_name)
class_names = metadata.thing_classes
id2item = {int(it["image_id"]): it for it in all_items}
# evaluator._predictions is a list of {image_id, instances: [coco-dets...]}.
# But we replaced it above with `all_predictions` (same shape). Group per image.
id2dets = {int(p["image_id"]): p.get("instances", []) for p in all_predictions}


def draw_gt(im, item):
    out = im.copy()
    for ann in item.get("annotations", []):
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cid = ann["category_id"]
        name = class_names[cid] if cid < len(class_names) else str(cid)
        cv2.putText(out, name, (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def draw_pred(im, dets, thresh=0.5):
    """``dets`` is a list of COCO-format detection dicts for one image."""
    from pycocotools import mask as mask_util
    out = im.copy()
    for d in dets:
        if d["score"] < thresh:
            continue
        x, y, w, h = [int(v) for v in d["bbox"]]
        cls = d["category_id"]
        name = class_names[cls] if cls < len(class_names) else str(cls)
        if "segmentation" in d:
            m = mask_util.decode(d["segmentation"])
            colored = np.zeros_like(out); colored[m > 0] = (0, 0, 255)
            out = cv2.addWeighted(out, 1.0, colored, 0.4, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 2)
        cv2.putText(out, f"{name} {d['score']:.2f}", (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    return out


sorted_ids = sorted(seen_ids)[: args.n_vis]
for iid in sorted_ids:
    item = id2item[iid]
    dets = id2dets[iid]
    im = cv2.imread(item["file_name"])
    gt = draw_gt(im, item)
    pr = draw_pred(im, dets, thresh=0.5)
    h = max(gt.shape[0], pr.shape[0])
    def pad(x): return cv2.copyMakeBorder(x, 0, h - x.shape[0], 0, 0,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
    panel = np.concatenate([pad(gt), pad(pr)], axis=1)
    cv2.putText(panel, "GT (green)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    n_kept = sum(1 for d in dets if d["score"] >= 0.5)
    cv2.putText(panel,
                f"W{meta.get('w_bits','?')}A{meta.get('a_bits','?')}  "
                f">=0.5: {n_kept}/{len(dets)}",
                (gt.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    cv2.imwrite(str(OUT / "vis" / f"{iid:012d}.jpg"), panel)
print(f"[merge] {len(sorted_ids)} vis saved to {OUT/'vis'}/")

# --- Optionally delete shard checkpoints ---
if args.rm_shards:
    for cp in shard_files:
        cp.unlink()
        print(f"[merge] removed {cp}")
