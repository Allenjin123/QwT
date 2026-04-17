"""Sanity test: swap nn.Linear / nn.Conv2d with Quant* wrappers on EVA-02 backbone,
keep all use_*_quant=False, run fp_smoke image. Output must match FP exactly.
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
from quant import quant_model_eva2, set_quant_state
from quant.quant_modules import QuantConv2d, QuantLinear, QuantMatMul

CFG_PATH = f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva02_large.py"
CKPT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva02_L_coco_det_sys_o365.pth"
IMG = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/coco/val2017/000000000139.jpg"


def main():
    cfg = LazyConfig.load(CFG_PATH)
    model = instantiate(cfg.model).eval().cuda()
    DetectionCheckpointer(model).load(CKPT)

    img = cv2.imread(IMG)
    img = cv2.resize(img, (1022, 1022))
    inp = torch.from_numpy(img).permute(2, 0, 1).float().cuda()
    batch = [{"image": inp, "height": img.shape[0], "width": img.shape[1]}]

    # FP baseline
    with torch.no_grad(), torch.cuda.amp.autocast():
        out_fp = model(batch)
    scores_fp = out_fp[0]["instances"].scores.cpu()

    # Swap with quant wrappers (quant OFF)
    quant_model_eva2(model, input_quant_params={"n_bits": 8}, weight_quant_params={"n_bits": 8})
    set_quant_state(model, False, False)

    n_qconv = sum(1 for m in model.modules() if isinstance(m, QuantConv2d))
    n_qlin = sum(1 for m in model.modules() if isinstance(m, QuantLinear))
    n_qmm = sum(1 for m in model.modules() if isinstance(m, QuantMatMul))
    print(f"[swap] QuantConv2d={n_qconv}  QuantLinear={n_qlin}  QuantMatMul={n_qmm}")

    with torch.no_grad(), torch.cuda.amp.autocast():
        out_q = model(batch)
    scores_q = out_q[0]["instances"].scores.cpu()

    diff = (scores_fp - scores_q).abs().max().item()
    print(f"[swap] max score diff = {diff:.2e}  (should be ~0)")
    assert diff < 1e-4, f"Quant-OFF swap changed scores by {diff}"
    print("[swap] PASS")

if __name__ == "__main__":
    main()
