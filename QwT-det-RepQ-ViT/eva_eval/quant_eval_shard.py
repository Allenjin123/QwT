"""One shard of a multi-GPU quantised eval run.

Each shard:
  1. Loads EVA FP model + weights
  2. Swaps backbone -> Quant*
  3. Runs calibration on the SAME first N_CALIB images (deterministic across shards
     so quantiser scales agree)
  4. Runs inference on its slice, saves D2-style predictions to results_<tag>/pred_shard<i>.pth
"""
import os, sys, time, argparse, math
from pathlib import Path
import torch

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
EVA_DET = "/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva_det"
sys.path.insert(0, EVA_DET)
sys.path.insert(0, "/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva_eval")

CFG_PATH = f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py"
CKPT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth"

p = argparse.ArgumentParser()
p.add_argument("--w-bits",  type=int, default=8)
p.add_argument("--a-bits",  type=int, default=8)
p.add_argument("--shard-idx", type=int, required=True)
p.add_argument("--n-shards",  type=int, required=True)
p.add_argument("--n-eval",  type=int, default=100)
p.add_argument("--n-calib", type=int, default=5)
p.add_argument("--tag", type=str, default=None)
p.add_argument("--reparam", action="store_true", help="RepQ-ViT scale reparam after calib")
p.add_argument("--qwt", action="store_true", help="install QwT compensation after calib")
p.add_argument("--qwt-n-samples", type=int, default=32)
p.add_argument("--qwt-start-block", type=int, default=0)
args = p.parse_args()

W, A = args.w_bits, args.a_bits
tag = args.tag or f"w{W}a{A}"
OUT = Path(__file__).parent / "results" / tag
OUT.mkdir(parents=True, exist_ok=True)
pred_file = OUT / f"pred_shard{args.shard_idx}.pth"

per = math.ceil(args.n_eval / args.n_shards)
lo = args.shard_idx * per
hi = min(lo + per, args.n_eval)
print(f"[shard {args.shard_idx}/{args.n_shards}] slice [{lo}:{hi}]  W{W}/A{A}  -> {pred_file.name}")

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog

from quant import (quant_model_eva, set_quant_state, scale_reparam_eva,
                   collapse_beit_like_qkv_bias, generate_compensation_model_eva)
from quant.quant_modules import QuantLinear, QuantConv2d, QuantMatMul

cfg = LazyConfig.load(CFG_PATH)
cfg.model.backbone.net.use_act_checkpoint = False
model = instantiate(cfg.model).eval().cuda()
DetectionCheckpointer(model).load(CKPT)

if args.reparam:
    n_collapsed = collapse_beit_like_qkv_bias(model.backbone.net)
    print(f"[shard {args.shard_idx}] collapsed beit_like qkv_bias on {n_collapsed} attentions")

quant_model_eva(
    model.backbone,
    input_quant_params=dict(n_bits=A),
    weight_quant_params=dict(n_bits=W, channel_wise=True),
)
print(f"[shard {args.shard_idx}] QuantLinear={sum(1 for m in model.modules() if isinstance(m, QuantLinear))} "
      f"QuantConv2d={sum(1 for m in model.modules() if isinstance(m, QuantConv2d))} "
      f"QuantMatMul={sum(1 for m in model.modules() if isinstance(m, QuantMatMul))}")

orig_name = cfg.dataloader.test.dataset.names
all_items = DatasetCatalog.get(orig_name)
subset = all_items[: args.n_eval]
sub_name = f"{orig_name}_first{args.n_eval}"
if sub_name not in DatasetCatalog.list():
    DatasetCatalog.register(sub_name, lambda s=subset: s)
    md = MetadataCatalog.get(orig_name).as_dict(); md.pop("name", None)
    MetadataCatalog.get(sub_name).set(**md)
cfg.dataloader.test.dataset.names = sub_name
test_loader = instantiate(cfg.dataloader.test)

set_quant_state(model, input_quant=True, weight_quant=True)
print(f"[shard {args.shard_idx}] calibrating on {args.n_calib} images...")
t0 = time.time()
with torch.no_grad():
    it = iter(test_loader)
    for _ in range(args.n_calib):
        model(next(it))
print(f"[shard {args.shard_idx}] calib {time.time()-t0:.1f}s")

if args.reparam:
    n_rp = scale_reparam_eva(model.backbone.net)
    print(f"[shard {args.shard_idx}] scale reparam on {n_rp} (LN, Linear) pairs")
    # weight quantizers were reset; one more forward to re-init them
    test_loader = instantiate(cfg.dataloader.test)
    t0 = time.time()
    with torch.no_grad():
        it = iter(test_loader)
        for _ in range(args.n_calib):
            model(next(it))
    print(f"[shard {args.shard_idx}] re-calib (post-reparam) {time.time()-t0:.1f}s")

if args.qwt:
    # Rebuild loader to get a fresh iterator for compensation calibration
    qwt_loader = instantiate(cfg.dataloader.test)
    t0 = time.time()
    report = generate_compensation_model_eva(
        model, qwt_loader, device=torch.device("cuda"),
        n_samples=args.qwt_n_samples, start_block=args.qwt_start_block,
        ridge=0.0, fwd_chunk=1,
        log=lambda s: print(f"[shard {args.shard_idx}] {s}"),
    )
    r2s = [r["r2"] for r in report]
    print(f"[shard {args.shard_idx}] QwT compensation done in {time.time()-t0:.1f}s "
          f"r2: min={min(r2s):.3f} mean={sum(r2s)/len(r2s):.3f} max={max(r2s):.3f}")
    # Compensation block adds quantized path; keep quant ON
    set_quant_state(model, input_quant=True, weight_quant=True)

test_loader = instantiate(cfg.dataloader.test)
predictions = []
t0 = time.time()
with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        if idx < lo: continue
        if idx >= hi: break
        out = model(batch)
        for inp, o in zip(batch, out):
            predictions.append({"image_id": inp["image_id"], "instances": o["instances"].to("cpu")})
dt = time.time() - t0
print(f"[shard {args.shard_idx}] inference on {hi-lo} imgs: {dt:.1f}s ({dt/max(hi-lo,1):.2f}s/img)")

torch.save({"predictions": predictions, "shard_idx": args.shard_idx,
            "n_shards": args.n_shards, "lo": lo, "hi": hi,
            "w_bits": W, "a_bits": A}, pred_file)
print(f"[shard {args.shard_idx}] saved {pred_file}")
