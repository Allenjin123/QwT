"""Merge shard predictions + run COCOEvaluator.

Usage: python quant_eval_merge.py --tag w8a8
Reads results_<tag>/pred_shard*.pth, combines, runs COCO eval restricted to the
union of image_ids seen, writes metrics.json + coco_instances_results.json + vis/.
"""
import os, sys, argparse, json
from pathlib import Path
import torch
import cv2
import numpy as np

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVA_DET = os.path.join(_THIS_DIR, "..", "eva_det")
sys.path.insert(0, EVA_DET)

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--n-vis", type=int, default=20)
args = p.parse_args()

OUT = Path(__file__).parent / "results" / args.tag
assert OUT.exists(), f"no such dir {OUT}"

from detectron2.config import LazyConfig
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation.coco_evaluation import instances_to_coco_json

cfg = LazyConfig.load(f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py")
orig_name = cfg.dataloader.test.dataset.names
all_items = DatasetCatalog.get(orig_name)

shard_files = sorted(OUT.glob("pred_shard*.pth"))
assert shard_files, f"no shard files in {OUT}"
print(f"[merge] loading {len(shard_files)} shards")
predictions, seen_ids, meta = [], set(), {}
for f in shard_files:
    d = torch.load(f, map_location="cpu", weights_only=False)
    fmt = d.get("format", "raw_v0")
    for pr in d["predictions"]:
        iid = int(pr["image_id"])
        seen_ids.add(iid)
        if fmt == "coco_v1":
            # New format: shard already RLE-encoded each image's predictions.
            predictions.extend(pr["coco"])
        else:
            # Legacy format: raw Instances object, convert here.
            predictions.extend(instances_to_coco_json(pr["instances"], iid))
    meta = {k: d[k] for k in ("w_bits", "a_bits", "n_shards")}
print(f"[merge] {len(predictions)} predictions, {len(seen_ids)} unique image_ids "
      f"(format={fmt})")

# Build evaluator; restrict coco api to images we predicted on
evaluator = COCOEvaluator(orig_name, tasks=("bbox", "segm"), distributed=False,
                          output_dir=str(OUT), max_dets_per_image=None)
coco = evaluator._coco_api
coco.imgs      = {k: v for k, v in coco.imgs.items() if k in seen_ids}
coco.anns      = {k: v for k, v in coco.anns.items() if v["image_id"] in seen_ids}
coco.imgToAnns = {k: v for k, v in coco.imgToAnns.items() if k in seen_ids}
coco.catToImgs = {c: [i for i in imgs if i in seen_ids] for c, imgs in coco.catToImgs.items()}

# `predictions` is already a flat list of COCO-format dicts (one per detection).
# Group them back by image so they match the shape COCOEvaluator._eval_predictions
# expects: a list of {"image_id": ..., "instances": [coco_det, coco_det, ...]}.
# Seed by_image from seen_ids so images with zero detections still appear (matches
# the legacy behaviour where every predicted image had an entry, possibly empty).
evaluator.reset()
by_image = {iid: [] for iid in seen_ids}
for det in predictions:
    by_image[int(det["image_id"])].append(det)
evaluator._predictions = [{"image_id": iid, "instances": dets}
                          for iid, dets in by_image.items()]
results = evaluator.evaluate()

print("[merge] Results:")
for task, m in results.items():
    print(f"  [{task}] AP={m.get('AP',0):.2f}  AP50={m.get('AP50',0):.2f}  AP75={m.get('AP75',0):.2f}")

metrics_out = {"n_eval": len(seen_ids), **meta, **results}
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"[merge] metrics.json -> {OUT/'metrics.json'}")

# FP comparison if available
fp_file = Path(__file__).parent / "results" / "fp100" / "metrics.json"
if fp_file.exists():
    with open(fp_file) as f: fp = json.load(f)
    print("[merge] Δ vs FP baseline:")
    for task in ("bbox", "segm"):
        if task in fp and task in results:
            d_ap   = results[task]["AP"]   - fp[task]["AP"]
            d_ap50 = results[task]["AP50"] - fp[task]["AP50"]
            print(f"    [{task}] ΔAP={d_ap:+.2f}  ΔAP50={d_ap50:+.2f}")

# --- visualisations (OpenCV) ---
(OUT / "vis").mkdir(exist_ok=True)
metadata = MetadataCatalog.get(orig_name)
class_names = metadata.thing_classes
id2item = {int(it["image_id"]): it for it in all_items}
# After the COCO-format refactor, predictions are flat per-detection dicts,
# not per-image Instances. Group them per-image for vis.
id2dets = {iid: by_image[iid] for iid in seen_ids}

def draw_gt(im, item):
    out = im.copy()
    for ann in item.get("annotations", []):
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cid = ann["category_id"]; name = class_names[cid] if cid < len(class_names) else str(cid)
        cv2.putText(out, name, (x, max(y-4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return out

def draw_pred(im, dets, thresh=0.5):
    """``dets`` is a list of COCO-format detection dicts for one image."""
    from pycocotools import mask as mask_util
    out = im.copy()
    for d in dets:
        if d["score"] < thresh: continue
        x, y, w, h = [int(v) for v in d["bbox"]]    # COCO bbox is xywh
        cls = d["category_id"]
        name = class_names[cls] if cls < len(class_names) else str(cls)
        if "segmentation" in d:
            m = mask_util.decode(d["segmentation"])
            colored = np.zeros_like(out); colored[m > 0] = (0, 0, 255)
            out = cv2.addWeighted(out, 1.0, colored, 0.4, 0)
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 165, 255), 2)
        cv2.putText(out, f"{name} {d['score']:.2f}", (x, max(y-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    return out

sorted_ids = sorted(seen_ids)[: args.n_vis]
for iid in sorted_ids:
    item = id2item[iid]; dets = id2dets[iid]
    im = cv2.imread(item["file_name"])
    gt = draw_gt(im, item); pr = draw_pred(im, dets, thresh=0.5)
    h = max(gt.shape[0], pr.shape[0])
    def pad(x): return cv2.copyMakeBorder(x, 0, h-x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    panel = np.concatenate([pad(gt), pad(pr)], axis=1)
    cv2.putText(panel, "GT (green)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    n_kept = sum(1 for d in dets if d["score"] >= 0.5)
    cv2.putText(panel, f"W{meta['w_bits']}A{meta['a_bits']}  >=0.5: {n_kept}/{len(dets)}",
                (gt.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
    cv2.imwrite(str(OUT/"vis"/f"{iid:012d}.jpg"), panel)
print(f"[merge] {len(sorted_ids)} vis saved to {OUT/'vis'}/")
