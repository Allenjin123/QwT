"""One shard of a multi-GPU quantised eval run.

Resumable, crash-safe variant modelled on ``det/run_coco_val.py``:

  1. Loads EVA FP model + weights.
  2. Optionally swaps backbone -> Quant*, calibrates, reparams, installs QwT.
  3. Runs inference on this shard's slice and feeds outputs straight into a
     ``COCOEvaluator`` (RLE-encodes masks per image).
  4. Periodically dumps the evaluator's per-image predictions to
     ``<tag>/shard_<id>/checkpoint.pt`` (atomic .tmp + replace), so a Ctrl-C
     or crash mid-eval can resume from the same image_id without redoing it.

Outputs (per shard, under ``<tag>/shard_<id>/``):
  * ``checkpoint.pt`` — {"config": ..., "predictions": [...COCO dicts...],
                          "image_ids_done": [...], "elapsed_seconds": ...}

The matching merge step (``quant_eval_merge.py``) globs all
``<tag>/shard_*/checkpoint.pt``, concatenates ``predictions`` lists into a
fresh evaluator, and runs ``evaluate()`` once to produce metrics.json +
coco_instances_results.json + instances_predictions.pth + vis/.

CLI essentials::

    python quant_eval_shard.py --tag w6a6_rp \\
        --w-bits 6 --a-bits 6 --reparam \\
        --num_shards 4 --shard_id 0 --n_eval 5000 \\
        [--resume]
"""
from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("DETECTRON2_DATASETS",
                      "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVA_DET = os.path.join(_THIS_DIR, "..", "eva_det")
sys.path.insert(0, EVA_DET)
sys.path.insert(0, _THIS_DIR)

CFG_PATH = f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py"
CKPT = os.environ.get(
    "EVA_CKPT",
    "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth")


def _atomic_save(state: dict, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _config_signature(args, eva_size: int, use_soft_nms: bool) -> dict:
    """Anything that, if changed mid-run, would invalidate prior predictions."""
    return {
        "w_bits": int(args.w_bits),
        "a_bits": int(args.a_bits),
        "reparam": bool(args.reparam),
        "qwt": bool(args.qwt),
        "qwt_n_samples": int(args.qwt_n_samples),
        "qwt_start_block": int(args.qwt_start_block),
        "no_quant": bool(args.no_quant),
        "n_calib": int(args.n_calib),
        "n_eval": int(args.n_eval),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "eva_size": int(eva_size),
        "use_soft_nms": bool(use_soft_nms),
    }


p = argparse.ArgumentParser()
p.add_argument("--w-bits", type=int, default=8)
p.add_argument("--a-bits", type=int, default=8)
p.add_argument("--shard_id", "--shard-idx", dest="shard_id", type=int, required=True,
               help="Which shard (0..num_shards-1) this process runs.")
p.add_argument("--num_shards", "--n-shards", dest="num_shards", type=int, required=True,
               help="Total number of shards in this run.")
p.add_argument("--n_eval", "--n-eval", dest="n_eval", type=int, default=100,
               help="Total # COCO val images to evaluate across all shards.")
p.add_argument("--n-calib", type=int, default=5)
p.add_argument("--tag", type=str, default=None)
p.add_argument("--reparam", action="store_true",
               help="RepQ-ViT scale reparam after calib")
p.add_argument("--qwt", action="store_true",
               help="install QwT compensation after calib")
p.add_argument("--qwt-n-samples", type=int, default=32)
p.add_argument("--qwt-start-block", type=int, default=0)
p.add_argument("--no-quant", action="store_true",
               help="Skip quant model wrapping, calibration, reparam and QwT. "
                    "Useful for an FP baseline through the same shard infra.")
p.add_argument("--resume", action="store_true",
               help="If shard_<id>/checkpoint.pt exists with matching config, "
                    "skip already-evaluated images and append new ones.")
p.add_argument("--save_every", type=int, default=50,
               help="Checkpoint cadence (images). Also flushes on Ctrl-C.")
args = p.parse_args()

W, A = args.w_bits, args.a_bits
tag = args.tag or f"w{W}a{A}"
TAG_DIR = Path(__file__).parent / "results" / tag
SHARD_DIR = TAG_DIR / f"shard_{args.shard_id}"
SHARD_DIR.mkdir(parents=True, exist_ok=True)
ckpt_path = SHARD_DIR / "checkpoint.pt"

# Even slice across shards (matches det/run_coco_val.py: first `rem` shards
# get one extra image so all `n_eval` images are covered without overlap).
base = args.n_eval // args.num_shards
rem = args.n_eval % args.num_shards
lo = args.shard_id * base + min(args.shard_id, rem)
shard_n = base + (1 if args.shard_id < rem else 0)
hi = lo + shard_n
print(f"[shard {args.shard_id}/{args.num_shards}] slice [{lo}:{hi})  n={shard_n}  "
      f"W{W}/A{A}  out={ckpt_path}")

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator

from quant import (quant_model_eva, set_quant_state, scale_reparam_eva,
                   collapse_beit_like_qkv_bias, generate_compensation_model_eva)
from quant.quant_modules import QuantLinear, QuantConv2d, QuantMatMul

# Square pad / ResizeShortestEdge override. 1280 = cfg default. Mirrors
# cascade_mask_rcnn_vitdet_eva_1536.py: keep ViT img_size at 1280 and switch
# interp_type to "beit" so trained pos embed gets bicubic-interpolated.
EVA_SIZE = int(os.environ.get("EVA_SIZE", "1280"))
assert EVA_SIZE % 256 == 0, f"EVA_SIZE={EVA_SIZE} must be a multiple of 256"
USE_SOFT_NMS = os.environ.get("USE_SOFT_NMS", "1") not in ("0", "false", "False", "")

cfg = LazyConfig.load(CFG_PATH)
cfg.model.backbone.net.use_act_checkpoint = False
if EVA_SIZE != 1280:
    cfg.model.backbone.square_pad = EVA_SIZE
    test_aug = cfg.dataloader.test.mapper.augmentations
    assert len(test_aug) == 1, "expected exactly one test-time augmentation"
    test_aug[0].short_edge_length = EVA_SIZE
    test_aug[0].max_size = EVA_SIZE
    if os.environ.get("BEIT_INTERP", "1") not in ("0", "false", "False", ""):
        cfg.model.backbone.net.interp_type = "beit"
        interp_msg = "interp_type=beit"
    else:
        interp_msg = "interp_type=vitdet (default, BEIT_INTERP=0)"
    print(f"[shard {args.shard_id}] EVA_SIZE={EVA_SIZE} "
          f"(square_pad+ResizeShortestEdge override; {interp_msg})")
if USE_SOFT_NMS:
    cfg.model.roi_heads.use_soft_nms = True
    print(f"[shard {args.shard_id}] use_soft_nms=True (linear, iou_thresh=0.3)")
model = instantiate(cfg.model).eval().cuda()
DetectionCheckpointer(model).load(CKPT)

if args.no_quant:
    print(f"[shard {args.shard_id}] --no-quant: skipping quant wrap / calib / "
          f"reparam / qwt (FP baseline through shard infra)")
else:
    if args.reparam:
        n_collapsed = collapse_beit_like_qkv_bias(model.backbone.net)
        print(f"[shard {args.shard_id}] collapsed beit_like qkv_bias on "
              f"{n_collapsed} attentions")

    quant_model_eva(
        model.backbone,
        input_quant_params=dict(n_bits=A),
        weight_quant_params=dict(n_bits=W, channel_wise=True),
    )
    print(f"[shard {args.shard_id}] "
          f"QuantLinear={sum(1 for m in model.modules() if isinstance(m, QuantLinear))} "
          f"QuantConv2d={sum(1 for m in model.modules() if isinstance(m, QuantConv2d))} "
          f"QuantMatMul={sum(1 for m in model.modules() if isinstance(m, QuantMatMul))}")

# Register the n_eval-image subset (used for both calib and inference).
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

if not args.no_quant:
    set_quant_state(model, input_quant=True, weight_quant=True)
    print(f"[shard {args.shard_id}] calibrating on {args.n_calib} images...")
    t0 = time.time()
    with torch.no_grad():
        it = iter(test_loader)
        for _ in range(args.n_calib):
            model(next(it))
    print(f"[shard {args.shard_id}] calib {time.time()-t0:.1f}s")

if args.reparam and not args.no_quant:
    n_rp = scale_reparam_eva(model.backbone.net)
    print(f"[shard {args.shard_id}] scale reparam on {n_rp} (LN, Linear) pairs")
    test_loader = instantiate(cfg.dataloader.test)
    t0 = time.time()
    with torch.no_grad():
        it = iter(test_loader)
        for _ in range(args.n_calib):
            model(next(it))
    print(f"[shard {args.shard_id}] re-calib (post-reparam) {time.time()-t0:.1f}s")

if args.qwt and not args.no_quant:
    qwt_loader = instantiate(cfg.dataloader.test)
    t0 = time.time()
    report = generate_compensation_model_eva(
        model, qwt_loader, device=torch.device("cuda"),
        n_samples=args.qwt_n_samples, start_block=args.qwt_start_block,
        ridge=0.0, fwd_chunk=1,
        log=lambda s: print(f"[shard {args.shard_id}] {s}"),
    )
    r2s = [r["r2"] for r in report]
    print(f"[shard {args.shard_id}] QwT compensation done in {time.time()-t0:.1f}s "
          f"r2: min={min(r2s):.3f} mean={sum(r2s)/len(r2s):.3f} max={max(r2s):.3f}")
    set_quant_state(model, input_quant=True, weight_quant=True)

# --- Set up the evaluator + restrict its COCO API to this shard's image set ---
# COCOEvaluator.process() converts each image's outputs to COCO format (RLE
# masks) on the fly and stashes them in evaluator._predictions. That list is
# what we checkpoint.
evaluator = COCOEvaluator(orig_name, tasks=("bbox", "segm"), distributed=False,
                          output_dir=str(SHARD_DIR), max_dets_per_image=None)
evaluator.reset()

cfg_sig = _config_signature(args, EVA_SIZE, USE_SOFT_NMS)

# --- Resume: load prior predictions if config matches ---
elapsed_prev = 0.0
done_ids: set[int] = set()
if args.resume and ckpt_path.exists():
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    prev_sig = ck.get("config", {})
    if prev_sig != cfg_sig:
        print(f"[resume] config mismatch — refusing to resume.")
        print(f"    prev: {prev_sig}")
        print(f"    curr: {cfg_sig}")
        sys.exit(2)
    evaluator._predictions = list(ck.get("predictions", []))
    done_ids = {int(p["image_id"]) for p in evaluator._predictions}
    elapsed_prev = float(ck.get("elapsed_seconds", 0.0))
    print(f"[resume] {len(done_ids)} images already done "
          f"({elapsed_prev:.1f}s prior wall time)")
elif ckpt_path.exists() and not args.resume:
    print(f"[warn] {ckpt_path} exists but --resume not set; OVERWRITING.")
    ckpt_path.unlink()

# Final pre-inference loader.
test_loader = instantiate(cfg.dataloader.test)


def _flush(elapsed_now: float):
    state = {
        "config": cfg_sig,
        "predictions": list(evaluator._predictions),
        "image_ids_done": sorted({int(p["image_id"])
                                  for p in evaluator._predictions}),
        "elapsed_seconds": elapsed_prev + elapsed_now,
    }
    _atomic_save(state, ckpt_path)


# Trap SIGINT: flush a checkpoint then exit cleanly.
interrupted = {"flag": False}


def _on_sigint(signum, frame):
    interrupted["flag"] = True
    print(f"\n[shard {args.shard_id}] SIGINT received; will checkpoint after "
          f"current image", flush=True)


prev_sigint = signal.signal(signal.SIGINT, _on_sigint)

t0 = time.time()
n_done_this_run = 0
try:
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx < lo:
                continue
            if idx >= hi:
                break
            iids = [int(b["image_id"]) for b in batch]
            if all(iid in done_ids for iid in iids):
                continue   # already accounted for in evaluator._predictions
            out = model(batch)
            evaluator.process(batch, out)
            n_done_this_run += len(batch)
            if n_done_this_run % args.save_every == 0:
                _flush(time.time() - t0)
                dt = time.time() - t0
                ips = n_done_this_run / max(dt, 1e-6)
                eta = (shard_n - len(done_ids) - n_done_this_run) / max(ips, 1e-6)
                print(f"[shard {args.shard_id}] "
                      f"{len(done_ids) + n_done_this_run}/{shard_n}  "
                      f"{dt:.0f}s  {ips:.2f} img/s  ETA {eta/60:.1f}min",
                      flush=True)
            if interrupted["flag"]:
                break
finally:
    _flush(time.time() - t0)
    signal.signal(signal.SIGINT, prev_sigint)

if interrupted["flag"]:
    print(f"[shard {args.shard_id}] interrupted; checkpoint saved at {ckpt_path}. "
          f"Re-run with --resume to continue.")
    sys.exit(130)

dt = time.time() - t0
total_done = len(evaluator._predictions)
print(f"[shard {args.shard_id}] inference on {n_done_this_run} new imgs: "
      f"{dt:.1f}s ({dt/max(n_done_this_run,1):.2f}s/img)  "
      f"shard total {total_done}/{shard_n}")
