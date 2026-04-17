"""Verify reparam math is transparent when quant is OFF.

Steps:
  (a) FP forward -> save scores
  (b) Collapse beit_like -> forward must == FP
  (c) Quant ON, calibrate -> per-channel scales populated
  (d) Reparam, then turn quant OFF
  (e) Forward -> must == FP (within FP precision)
"""
import os, sys
import torch, cv2

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVA_DET = os.path.join(_THIS_DIR, "..", "eva_det")
sys.path.insert(0, EVA_DET)
sys.path.insert(0, _THIS_DIR)

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from quant import quant_model_eva, set_quant_state, scale_reparam_eva, collapse_beit_like_qkv_bias

CFG = f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py"
CKPT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth"
IMG = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/coco/val2017/000000000139.jpg"

cfg = LazyConfig.load(CFG)
cfg.model.backbone.net.use_act_checkpoint = False
model = instantiate(cfg.model).eval().cuda()
DetectionCheckpointer(model).load(CKPT)

im = cv2.imread(IMG)
img_t = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
inp = [{"image": img_t, "height": im.shape[0], "width": im.shape[1]}]

def fwd():
    with torch.no_grad():
        return model(inp)[0]["instances"].to("cpu")

print("[a] FP forward")
fp = fwd()
print(f"    num={len(fp)}  top3={fp.scores[:3].tolist()}")

print("[b] Collapse beit_like_qkv_bias and re-run FP -- must match")
n = collapse_beit_like_qkv_bias(model.backbone.net)
print(f"    collapsed {n} attentions")
fp2 = fwd()
print(f"    num={len(fp2)}  top3={fp2.scores[:3].tolist()}")
diff_b = (fp.scores - fp2.scores).abs().max().item() if len(fp)==len(fp2) else float('nan')
print(f"    max|Δscore| (b vs a) = {diff_b:.2e}")

print("[c] Swap to Quant*, calibrate (channel-wise on qkv/fc1)")
quant_model_eva(model.backbone, input_quant_params=dict(n_bits=8),
                weight_quant_params=dict(n_bits=8, channel_wise=True))
set_quant_state(model, input_quant=True, weight_quant=True)
fwd()  # one calib pass

print("[d] Apply reparam, then quant OFF")
n_rp = scale_reparam_eva(model.backbone.net)
print(f"    reparam'd {n_rp} pairs")
set_quant_state(model, input_quant=False, weight_quant=False)

print("[e] Forward with quant OFF -- must match FP within precision")
fp3 = fwd()
print(f"    num={len(fp3)}  top3={fp3.scores[:3].tolist()}")
diff_e = (fp.scores - fp3.scores).abs().max().item() if len(fp)==len(fp3) else float('nan')
print(f"    max|Δscore| (e vs a) = {diff_e:.2e}")
print("[OK] reparam math transparent" if diff_e < 1e-2 else "[FAIL] reparam math broken")
