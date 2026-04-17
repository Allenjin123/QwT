"""Minimal FP forward test: EVA-02 ViTDet + Cascade Mask RCNN on 1 COCO image."""
import os, sys, time
import torch
import cv2
import numpy as np

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVA_DET = os.path.join(_THIS_DIR, "..", "eva_det")
CFG_PATH = f"{EVA_DET}/projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py"
CKPT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva02_L_coco_det_sys_o365.pth"
IMG = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/coco/val2017/000000000139.jpg"

sys.path.insert(0, EVA_DET)

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

def main():
    cfg = LazyConfig.load(CFG_PATH)
    model = instantiate(cfg.model)
    model.eval().cuda()

    ckpt = DetectionCheckpointer(model)
    ckpt.load(CKPT)

    img = cv2.imread(IMG)
    img = cv2.resize(img, (1536, 1536))
    inp = torch.from_numpy(img).permute(2, 0, 1).float().cuda()
    batch = [{"image": inp, "height": img.shape[0], "width": img.shape[1]}]

    with torch.no_grad(), torch.cuda.amp.autocast():
        t0 = time.time()
        out = model(batch)
        torch.cuda.synchronize()
        dt = time.time() - t0

    instances = out[0]["instances"]
    print(f"[fp_smoke] {len(instances)} detections in {dt:.2f}s")
    scores = instances.scores.cpu()
    if len(scores):
        print(f"  top-5 scores: {scores[:5].tolist()}")
        print(f"  classes:      {instances.pred_classes[:5].cpu().tolist()}")

if __name__ == "__main__":
    main()
