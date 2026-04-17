"""Verify reparam math is transparent when quant is OFF.

Steps:
  (a) FP forward -> save scores
  (b) Collapse q/v bias -> forward must == FP
  (c) Quant ON, calibrate -> per-channel scales populated
  (d) Reparam, then turn quant OFF
  (e) Forward -> must == FP (within FP precision)
"""
import os, sys
import torch, cv2

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
EVA_DET = "/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva2/eva_det"
sys.path.insert(0, EVA_DET)
sys.path.insert(0, "/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva2/eva_eval")

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from quant import quant_model_eva2, set_quant_state, scale_reparam_eva2, collapse_qkv_bias

CFG_PATH = f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva02_large.py"
CKPT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva02_L_coco_det_sys_o365.pth"
IMG = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/coco/val2017/000000000139.jpg"


def build_batch():
    img = cv2.imread(IMG)
    img = cv2.resize(img, (1022, 1022))
    inp = torch.from_numpy(img).permute(2, 0, 1).float().cuda()
    return [{"image": inp, "height": img.shape[0], "width": img.shape[1]}]


@torch.no_grad()
def get_scores(model, batch):
    with torch.cuda.amp.autocast():
        out = model(batch)
    return out[0]["instances"].scores.cpu()


def main():
    cfg = LazyConfig.load(CFG_PATH)
    model = instantiate(cfg.model).eval().cuda()
    DetectionCheckpointer(model).load(CKPT)
    batch = build_batch()

    # (a) FP baseline
    s_fp = get_scores(model, batch)
    print(f"(a) FP: {len(s_fp)} dets")

    # (b) Collapse bias
    n_col = collapse_qkv_bias(model)
    s_col = get_scores(model, batch)
    d = (s_fp - s_col).abs().max().item()
    print(f"(b) collapse {n_col} blocks, max diff = {d:.2e}")
    assert d < 1e-4

    # (c) Quant swap + calibrate
    quant_model_eva2(model, {"n_bits": 8}, {"n_bits": 8})
    set_quant_state(model, True, True)
    _ = get_scores(model, batch)  # calibration forward
    print("(c) calibrated")

    # (d) Reparam then quant OFF
    n_rp = scale_reparam_eva2(model.backbone.net)
    set_quant_state(model, False, False)
    s_rp = get_scores(model, batch)
    d = (s_fp - s_rp).abs().max().item()
    print(f"(d) reparam {n_rp} pairs, quant OFF, max diff = {d:.2e}")
    assert d < 1e-3, f"reparam changed scores by {d}"
    print("PASS")

if __name__ == "__main__":
    main()
