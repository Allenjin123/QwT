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
EVA_DET = "/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva1/eva_det"
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
    predictions.extend(d["predictions"])
    meta = {k: d[k] for k in ("w_bits", "a_bits", "n_shards")}
    for pr in d["predictions"]:
        seen_ids.add(int(pr["image_id"]))
print(f"[merge] {len(predictions)} predictions, {len(seen_ids)} unique image_ids")

# Build evaluator; restrict coco api to images we predicted on
evaluator = COCOEvaluator(orig_name, tasks=("bbox", "segm"), distributed=False,
                          output_dir=str(OUT), max_dets_per_image=None)
coco = evaluator._coco_api
coco.imgs      = {k: v for k, v in coco.imgs.items() if k in seen_ids}
coco.anns      = {k: v for k, v in coco.anns.items() if v["image_id"] in seen_ids}
coco.imgToAnns = {k: v for k, v in coco.imgToAnns.items() if k in seen_ids}
coco.catToImgs = {c: [i for i in imgs if i in seen_ids] for c, imgs in coco.catToImgs.items()}

# COCOEvaluator._predictions expects list of dicts with "instances" as raw Instances object
# and converts to COCO format in _eval_predictions via instances_to_coco_json
evaluator.reset()
# Convert raw Instances -> COCO dicts (what COCOEvaluator.process() would have done)
coco_preds = [{"image_id": int(pr["image_id"]),
               "instances": instances_to_coco_json(pr["instances"], int(pr["image_id"]))}
              for pr in predictions]
evaluator._predictions = coco_preds
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
id2pred = {int(pr["image_id"]): pr["instances"] for pr in predictions}

def draw_gt(im, item):
    out = im.copy()
    for ann in item.get("annotations", []):
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cid = ann["category_id"]; name = class_names[cid] if cid < len(class_names) else str(cid)
        cv2.putText(out, name, (x, max(y-4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return out

def draw_pred(im, inst, thresh=0.5):
    out = im.copy()
    scores = inst.scores.numpy(); boxes = inst.pred_boxes.tensor.numpy(); classes = inst.pred_classes.numpy()
    has_mask = inst.has("pred_masks"); masks = inst.pred_masks.numpy() if has_mask else None
    for i in range(len(inst)):
        if scores[i] < thresh: continue
        x1,y1,x2,y2 = [int(v) for v in boxes[i]]
        cls = classes[i]; name = class_names[cls] if cls < len(class_names) else str(cls)
        if has_mask:
            m = masks[i].astype(np.uint8); colored = np.zeros_like(out); colored[m>0] = (0, 0, 255)
            out = cv2.addWeighted(out, 1.0, colored, 0.4, 0)
        cv2.rectangle(out, (x1,y1), (x2,y2), (0, 165, 255), 2)
        cv2.putText(out, f"{name} {scores[i]:.2f}", (x1, max(y1-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    return out

sorted_ids = sorted(seen_ids)[: args.n_vis]
for iid in sorted_ids:
    item = id2item[iid]; inst = id2pred[iid]
    im = cv2.imread(item["file_name"])
    gt = draw_gt(im, item); pr = draw_pred(im, inst, thresh=0.5)
    h = max(gt.shape[0], pr.shape[0])
    def pad(x): return cv2.copyMakeBorder(x, 0, h-x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    panel = np.concatenate([pad(gt), pad(pr)], axis=1)
    cv2.putText(panel, "GT (green)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    n_kept = int((inst.scores >= 0.5).sum())
    cv2.putText(panel, f"W{meta['w_bits']}A{meta['a_bits']}  >=0.5: {n_kept}/{len(inst)}",
                (gt.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
    cv2.imwrite(str(OUT/"vis"/f"{iid:012d}.jpg"), panel)
print(f"[merge] {len(sorted_ids)} vis saved to {OUT/'vis'}/")
