"""Minimal FP forward test: EVA-01 ViTDet + Cascade Mask RCNN on 1 COCO image."""
import os, sys, time
import torch
import cv2
import numpy as np

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")

EVA_DET = "/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva1/eva_det"
CFG_PATH = f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py"
CKPT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth"
IMG = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/coco/val2017/000000000139.jpg"

sys.path.insert(0, EVA_DET)

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

print("[1] Loading config:", CFG_PATH)
cfg = LazyConfig.load(CFG_PATH)
cfg.model.backbone.net.use_act_checkpoint = False

print(f"[2] Building model (ViT-g depth={cfg.model.backbone.net.depth} embed_dim={cfg.model.backbone.net.embed_dim})")
model = instantiate(cfg.model)
model.eval().cuda()
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"    params: {n_params:.1f}M")

print(f"[3] Loading weights: {CKPT}")
t0 = time.time()
DetectionCheckpointer(model).load(CKPT)
print(f"    loaded in {time.time()-t0:.1f}s")

print(f"[4] Reading image: {IMG}")
im = cv2.imread(IMG)
print(f"    shape: {im.shape}")

print("[5] Forward (this may take ~10-30s for ViT-g @ 1280)")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad():
    img_t = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
    out = model([{"image": img_t, "height": im.shape[0], "width": im.shape[1]}])
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1e9
inst = out[0]["instances"]
print(f"    forward {dt:.1f}s  peak_mem={peak:.1f} GB  num_instances={len(inst)}")
if len(inst):
    print(f"    top5 scores: {inst.scores[:5].tolist()}")
    print(f"    top5 classes: {inst.pred_classes[:5].tolist()}")
print("[OK] FP smoke test passed")
