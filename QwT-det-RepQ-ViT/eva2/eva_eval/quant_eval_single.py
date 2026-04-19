"""Run a single bits config (reparam+qwt) on the GPU visible to this process.

Usage:
    CUDA_VISIBLE_DEVICES=0 python quant_eval_single.py 8
    CUDA_VISIBLE_DEVICES=1 python quant_eval_single.py 6
"""
import os, sys, time, json, copy
from pathlib import Path
import torch

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVA_DET = os.path.join(_THIS_DIR, "..", "eva_det")
sys.path.insert(0, EVA_DET)
sys.path.insert(0, "/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva2/eva_eval")

CFG_PATH = f"{EVA_DET}/projects/ViTDet/configs/eva2_o365_to_coco/eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py"
CKPT = "/scratch/nbleier_owned_root/nbleier_owned1/shared_data/pretrained/eva02_L_coco_det_sys_o365.pth"
N_EVAL = 100
N_CALIB = 32

from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
from quant import (quant_model_eva2, set_quant_state, scale_reparam_eva2,
                   collapse_qkv_bias, generate_compensation_model_eva)


def build_eval(cfg_template, n_eval, tag):
    cfg = copy.deepcopy(cfg_template)
    orig_name = cfg.dataloader.test.dataset.names
    all_items = DatasetCatalog.get(orig_name)
    subset = all_items[:n_eval]
    sub_name = f"{orig_name}_first{n_eval}_{tag}"
    if sub_name in DatasetCatalog.list():
        DatasetCatalog.remove(sub_name)
        MetadataCatalog.remove(sub_name)
    DatasetCatalog.register(sub_name, lambda: subset)
    src_meta = MetadataCatalog.get(orig_name).as_dict()
    src_meta.pop("name", None)
    MetadataCatalog.get(sub_name).set(**src_meta)
    cfg.dataloader.test.dataset.names = sub_name
    test_loader = instantiate(cfg.dataloader.test)
    evaluator = COCOEvaluator(orig_name, tasks=("bbox", "segm"), distributed=False,
                              output_dir=f"/tmp/eva2_quant_eval_{tag}")
    subset_img_ids = set(int(it["image_id"]) for it in subset)
    coco = evaluator._coco_api
    coco.imgs = {k: v for k, v in coco.imgs.items() if k in subset_img_ids}
    coco.anns = {k: v for k, v in coco.anns.items() if v["image_id"] in subset_img_ids}
    coco.imgToAnns = {k: v for k, v in coco.imgToAnns.items() if k in subset_img_ids}
    coco.catToImgs = {c: [i for i in imgs if i in subset_img_ids]
                      for c, imgs in coco.catToImgs.items()}
    return test_loader, evaluator, subset


def calib_forward(model, subset, n_calib):
    import cv2
    with torch.no_grad():
        for i, item in enumerate(subset[:n_calib]):
            im = cv2.imread(item["file_name"])
            img_t = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
            model([{"image": img_t.cuda(), "height": im.shape[0], "width": im.shape[1]}])
            if (i + 1) % 8 == 0:
                print(f"  calib {i+1}/{n_calib}", flush=True)


def run(bits, mode):
    """mode: 'rp' (reparam only) or 'rp_qwt' (reparam + qwt)"""
    tag = f"w{bits}a{bits}_{mode}"
    print(f"\n{'='*60}\n  {tag}\n{'='*60}", flush=True)

    cfg_template = LazyConfig.load(CFG_PATH)
    _, _, subset = build_eval(cfg_template, N_EVAL, f"{tag}_prep")

    cfg = copy.deepcopy(cfg_template)
    cfg.model.backbone.net.xattn = False
    cfg.model.backbone.net.use_act_checkpoint = False
    model = instantiate(cfg.model).eval().cuda()
    DetectionCheckpointer(model).load(CKPT)

    n_col = collapse_qkv_bias(model)
    print(f"[collapse] {n_col} blocks", flush=True)

    quant_model_eva2(model, {"n_bits": bits}, {"n_bits": bits, "channel_wise": True})
    set_quant_state(model, True, True)

    print(f"[calib] {N_CALIB} images...", flush=True)
    calib_forward(model, subset, N_CALIB)
    print("[calib] done", flush=True)

    n_rp = scale_reparam_eva2(model.backbone.net)
    print(f"[reparam] {n_rp} pairs", flush=True)
    print(f"[re-calib] {N_CALIB} images...", flush=True)
    calib_forward(model, subset, N_CALIB)
    print("[re-calib] done", flush=True)

    if mode == "rp_qwt":
        qwt_loader, _, _ = build_eval(cfg_template, N_EVAL, f"{tag}_qwt")
        t0 = time.time()
        report = generate_compensation_model_eva(
            model, qwt_loader, device=torch.device("cuda"),
            n_samples=32, start_block=0, ridge=0.0, fwd_chunk=1,
            log=lambda s: print(f"  {s}", flush=True),
        )
        r2s = [r["r2"] for r in report]
        print(f"[qwt] done in {time.time()-t0:.1f}s  "
              f"r2: min={min(r2s):.3f} mean={sum(r2s)/len(r2s):.3f} max={max(r2s):.3f}",
              flush=True)
        set_quant_state(model, True, True)

    test_loader, evaluator, _ = build_eval(cfg_template, N_EVAL, f"{tag}_eval")
    print(f"[eval] {N_EVAL} images...", flush=True)
    t0 = time.time()
    results = inference_on_dataset(model, test_loader, evaluator)
    dt = time.time() - t0
    print(f"[eval] done in {dt:.1f}s", flush=True)

    for task, m in results.items():
        print(f"  [{task}] AP={m.get('AP',0):.2f}  AP50={m.get('AP50',0):.2f}  "
              f"AP75={m.get('AP75',0):.2f}", flush=True)

    OUT = Path(__file__).parent / "results" / "quant_rp_qwt"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{tag}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {OUT/f'{tag}.json'}", flush=True)


if __name__ == "__main__":
    bits = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "rp_qwt"
    run(bits, mode)
