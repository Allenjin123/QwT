"""FP eval at a configurable input size on first N COCO val images.

Usage:
    python fp_eval_size.py --size 1024 --n 100
    python fp_eval_size.py --size 768  --n 100

Notes on size:
- patch_size=16, window_size=16 (in tokens) -> input must be a multiple of 256.
- Allowed sizes: 256, 512, 768, 1024, 1280 (trained), 1536...
- We keep model.backbone.net.img_size=1280 (matches checkpoint rel_pos params);
  rel_pos and abs pos_embed are interpolated at runtime.
- We change square_pad and the test ResizeShortestEdge to the chosen size.
"""
import argparse, os, sys, time, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=int, required=True, help="input square size, multiple of 256")
ap.add_argument("--n", type=int, default=100)
args = ap.parse_args()

assert args.size % 256 == 0, f"size {args.size} must be multiple of 256 (patch*window=16*16)"

import torch

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVA_DET = os.path.join(_THIS_DIR, "..", "eva_det")
sys.path.insert(0, EVA_DET)

CFG_PATH = f"{EVA_DET}/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vitdet_eva.py"
CKPT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva_coco_det.pth"
OUT = Path(__file__).parent / "results" / f"fp{args.n}_sz{args.size}"
OUT.mkdir(parents=True, exist_ok=True)

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog

print(f"[1] Loading config (target size={args.size})")
cfg = LazyConfig.load(CFG_PATH)
cfg.model.backbone.net.use_act_checkpoint = False
cfg.model.backbone.square_pad = args.size
# leave net.img_size=1280 to match checkpoint rel_pos shape; runtime interp handles it.
# update test resize to chosen size
test_aug = cfg.dataloader.test.mapper.augmentations
assert len(test_aug) == 1, f"expected 1 test aug, got {len(test_aug)}"
test_aug[0].short_edge_length = args.size
test_aug[0].max_size = args.size

print("[2] Building model")
model = instantiate(cfg.model)
model.eval().cuda()
print(f"    params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

print("[3] Loading weights")
DetectionCheckpointer(model).load(CKPT)

print(f"[4] Building test loader (first {args.n} of coco_2017_val)")
orig_name = cfg.dataloader.test.dataset.names
all_items = DatasetCatalog.get(orig_name)
subset = all_items[:args.n]
sub_name = f"{orig_name}_first{args.n}_sz{args.size}"
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

print(f"[6] Running inference on {args.n} images @ size={args.size}")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
results = inference_on_dataset(model, test_loader, evaluator)
dt = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"    done in {dt:.1f}s ({dt/args.n:.2f}s/img)  peak_mem={peak:.1f} GB")

metrics_out = {"size": args.size, "n_eval": args.n, "eval_seconds": dt,
               "peak_mem_gb": peak, **results}
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics_out, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
print(f"[7] Summary (size={args.size}):")
for task, m in results.items():
    print(f"    [{task}] AP={m.get('AP',0):.2f}  AP50={m.get('AP50',0):.2f}  AP75={m.get('AP75',0):.2f}")
print(f"[OK] metrics -> {OUT/'metrics.json'}")
