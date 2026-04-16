"""Sanity test: swap nn.Linear / nn.Conv2d with Quant* wrappers on EVA-01 backbone,
keep all use_*_quant=False, run fp_smoke image. Output must match FP exactly.
"""
import os, sys
import torch, cv2

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
EVA_DET = "/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva_det"
sys.path.insert(0, EVA_DET)
sys.path.insert(0, "/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva_eval")

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from quant import quant_model_eva, set_quant_state
from quant.quant_modules import QuantConv2d, QuantLinear, QuantMatMul

CFG = f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py"
CKPT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth"
IMG = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/coco/val2017/000000000139.jpg"

cfg = LazyConfig.load(CFG)
cfg.model.backbone.net.use_act_checkpoint = False

print("[1] Building + loading FP model, running forward")
model = instantiate(cfg.model).eval().cuda()
DetectionCheckpointer(model).load(CKPT)
im = cv2.imread(IMG)
img_t = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
inp = [{"image": img_t, "height": im.shape[0], "width": im.shape[1]}]

with torch.no_grad():
    pred_fp = model(inp)[0]["instances"].to("cpu")
print(f"    FP: num={len(pred_fp)}  top3_scores={pred_fp.scores[:3].tolist()}")

print("[2] Swapping backbone nn.Linear / nn.Conv2d -> Quant* (keeping quant OFF)")
# Only swap inside backbone to limit surface area for this sanity check
n_before = sum(1 for m in model.modules() if isinstance(m, (QuantConv2d, QuantLinear)))
quant_model_eva(
    model.backbone,
    input_quant_params=dict(n_bits=8),
    weight_quant_params=dict(n_bits=8, channel_wise=True),
)
n_after = sum(1 for m in model.modules() if isinstance(m, (QuantConv2d, QuantLinear)))
n_mm   = sum(1 for m in model.modules() if isinstance(m, QuantMatMul))
n_mm_named = sum(1 for n, _ in model.named_modules() if n.endswith(".matmul1") or n.endswith(".matmul2"))
print(f"    QuantLinear+QuantConv2d: {n_before} -> {n_after}")
print(f"    QuantMatMul: {n_mm}  (named matmul1/2 count: {n_mm_named})")

set_quant_state(model, input_quant=False, weight_quant=False)

with torch.no_grad():
    pred_q = model(inp)[0]["instances"].to("cpu")
print(f"    swapped (quant OFF): num={len(pred_q)}  top3_scores={pred_q.scores[:3].tolist()}")

# Compare
same_num = len(pred_fp) == len(pred_q)
if same_num:
    max_score_diff = (pred_fp.scores - pred_q.scores).abs().max().item()
    max_box_diff = (pred_fp.pred_boxes.tensor - pred_q.pred_boxes.tensor).abs().max().item()
    print(f"[3] diff:  scores max|Δ|={max_score_diff:.2e}   boxes max|Δ|={max_box_diff:.2e}")
    ok = max_score_diff < 1e-5 and max_box_diff < 1e-4
else:
    print(f"[3] num_instances differ: {len(pred_fp)} vs {len(pred_q)}")
    ok = False

print("[OK] swap transparent" if ok else "[FAIL] swap changed output")
