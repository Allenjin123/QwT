"""FP eval on first 100 COCO val images.

Outputs under ./results_fp100/:
  metrics.json                - AP / AP50 / AP75 / per-class
  coco_instances_results.json - raw predictions (pycocotools format)
  vis/<image_id>.jpg          - GT vs prediction side-by-side (first N_VIS imgs)
"""
import os, sys, time, json
from pathlib import Path
import torch
import cv2
import numpy as np

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVA_DET = os.path.join(_THIS_DIR, "..", "eva_det")
sys.path.insert(0, EVA_DET)

CFG_PATH = f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py"
CKPT = os.environ.get(
    "EVA_CKPT",
    "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth")
N_EVAL = int(os.environ.get("N_EVAL", "100"))
N_VIS = int(os.environ.get("N_VIS", "20"))
# Square pad / ResizeShortestEdge override. 1280 = cfg default. Must be a
# multiple of patch_size * window_size = 256. Mirrors the official
# cascade_mask_rcnn_vitdet_eva_1536.py trick: keep ViT img_size at 1280 (so
# pos embed shape stays as trained) and switch interp_type to "beit" for
# on-the-fly pos embed interpolation at the new resolution.
EVA_SIZE = int(os.environ.get("EVA_SIZE", "1280"))
assert EVA_SIZE % 256 == 0, f"EVA_SIZE={EVA_SIZE} must be a multiple of 256"
_size_tag = "" if EVA_SIZE == 1280 else f"_sz{EVA_SIZE}"
# When the user explicitly asks for hard NMS via USE_SOFT_NMS=0, suffix the
# output dir so it doesn't collide with soft-NMS runs at the same size/n_eval.
_nms_tag = "" if os.environ.get("USE_SOFT_NMS", "1") not in ("0", "false", "False", "") else "_hardnms"
OUT = Path(__file__).parent / "results" / f"fp{N_EVAL}{_size_tag}{_nms_tag}"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "vis").mkdir(exist_ok=True)

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

USE_SOFT_NMS = os.environ.get("USE_SOFT_NMS", "1") not in ("0", "false", "False", "")
print(f"[1] Loading config (EVA_SIZE={EVA_SIZE} use_soft_nms={USE_SOFT_NMS})")
cfg = LazyConfig.load(CFG_PATH)
cfg.model.backbone.net.use_act_checkpoint = False
if EVA_SIZE != 1280:
    cfg.model.backbone.square_pad = EVA_SIZE
    cfg.model.backbone.net.interp_type = "beit"
    test_aug = cfg.dataloader.test.mapper.augmentations
    assert len(test_aug) == 1, "expected exactly one test-time augmentation"
    test_aug[0].short_edge_length = EVA_SIZE
    test_aug[0].max_size = EVA_SIZE
if USE_SOFT_NMS:
    cfg.model.roi_heads.use_soft_nms = True

print("[2] Building model")
model = instantiate(cfg.model)
model.eval().cuda()
print(f"    params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

print("[3] Loading weights")
DetectionCheckpointer(model).load(CKPT)

print("[4] Building test loader (first 100 of coco_2017_val)")
orig_name = cfg.dataloader.test.dataset.names
all_items = DatasetCatalog.get(orig_name)
subset = all_items[:N_EVAL]
sub_name = f"{orig_name}_first{N_EVAL}"
if sub_name in DatasetCatalog.list():
    DatasetCatalog.remove(sub_name); MetadataCatalog.remove(sub_name)
DatasetCatalog.register(sub_name, lambda: subset)
src_meta = MetadataCatalog.get(orig_name).as_dict(); src_meta.pop("name", None)
MetadataCatalog.get(sub_name).set(**src_meta)
cfg.dataloader.test.dataset.names = sub_name
test_loader = instantiate(cfg.dataloader.test)

print("[5] Building COCOEvaluator (imgIds restricted to subset)")
evaluator = COCOEvaluator(orig_name, tasks=("bbox", "segm"), distributed=False,
                          output_dir=str(OUT), max_dets_per_image=None)
subset_img_ids = set(int(it["image_id"]) for it in subset)
coco = evaluator._coco_api
coco.imgs = {k: v for k, v in coco.imgs.items() if k in subset_img_ids}
coco.anns = {k: v for k, v in coco.anns.items() if v["image_id"] in subset_img_ids}
coco.imgToAnns = {k: v for k, v in coco.imgToAnns.items() if k in subset_img_ids}
coco.catToImgs = {c: [i for i in imgs if i in subset_img_ids] for c, imgs in coco.catToImgs.items()}

print(f"[6] Running inference on {N_EVAL} images")
t0 = time.time()
results = inference_on_dataset(model, test_loader, evaluator)
dt = time.time() - t0
print(f"    done in {dt:.1f}s ({dt/N_EVAL:.2f}s/img)")

metrics_out = {"n_eval": N_EVAL, "eval_seconds": dt, **results}
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2)
print(f"[7] metrics.json saved. Summary:")
for task, m in results.items():
    print(f"    [{task}] AP={m.get('AP',0):.2f}  AP50={m.get('AP50',0):.2f}  AP75={m.get('AP75',0):.2f}")

print(f"[8] Rendering {N_VIS} visualisations (OpenCV-only, no matplotlib)")
metadata = MetadataCatalog.get(orig_name)
class_names = metadata.thing_classes

def draw_gt(im, item):
    out = im.copy()
    for ann in item.get("annotations", []):
        x, y, w, h = [int(v) for v in ann["bbox"]]
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cid = ann["category_id"]
        name = class_names[cid] if cid < len(class_names) else str(cid)
        cv2.putText(out, name, (x, max(y-4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return out

def draw_pred(im, inst, thresh=0.5):
    out = im.copy()
    scores = inst.scores.numpy()
    boxes = inst.pred_boxes.tensor.numpy()
    classes = inst.pred_classes.numpy()
    has_mask = inst.has("pred_masks")
    masks = inst.pred_masks.numpy() if has_mask else None
    for i in range(len(inst)):
        if scores[i] < thresh: continue
        x1,y1,x2,y2 = [int(v) for v in boxes[i]]
        cls = classes[i]; name = class_names[cls] if cls < len(class_names) else str(cls)
        if has_mask:
            m = masks[i].astype(np.uint8)
            colored = np.zeros_like(out); colored[m>0] = (0, 0, 255)
            out = cv2.addWeighted(out, 1.0, colored, 0.4, 0)
        cv2.rectangle(out, (x1,y1), (x2,y2), (0, 165, 255), 2)
        cv2.putText(out, f"{name} {scores[i]:.2f}", (x1, max(y1-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    return out

with torch.no_grad():
    for item in subset[:N_VIS]:
        im = cv2.imread(item["file_name"])
        img_t = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
        pred = model([{"image": img_t, "height": im.shape[0], "width": im.shape[1]}])[0]
        inst = pred["instances"].to("cpu")

        gt_panel   = draw_gt(im, item)
        pred_panel = draw_pred(im, inst, thresh=0.5)

        h = max(gt_panel.shape[0], pred_panel.shape[0])
        def pad(x): return cv2.copyMakeBorder(x, 0, h-x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        panel = np.concatenate([pad(gt_panel), pad(pred_panel)], axis=1)
        cv2.putText(panel, "GT (green)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        n_kept = int((inst.scores >= 0.5).sum())
        cv2.putText(panel, f"PRED score>=0.5: {n_kept}/{len(inst)}",
                    (gt_panel.shape[1]+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
        cv2.imwrite(str(OUT/"vis"/f"{item['image_id']:012d}.jpg"), panel)

print(f"[OK] Done.")
print(f"     cat  {OUT/'metrics.json'}")
print(f"     ls   {OUT/'vis'}/")
