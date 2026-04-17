"""Quantised eval on first 100 COCO val images — EVA-02-L, NO reparam.
Runs w8a8, w6a6, w4a4 sequentially.
"""
import os, sys, time, json, copy
from pathlib import Path
import torch

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
EVA_DET = "/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva2/eva_det"
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
from quant import quant_model_eva2, set_quant_state, collapse_qkv_bias


def build_subset_loader(cfg, n_eval):
    orig_name = cfg.dataloader.test.dataset.names
    all_items = DatasetCatalog.get(orig_name)
    subset = all_items[:n_eval]
    sub_name = f"{orig_name}_first{n_eval}"
    if sub_name in DatasetCatalog.list():
        DatasetCatalog.remove(sub_name)
        MetadataCatalog.remove(sub_name)
    DatasetCatalog.register(sub_name, lambda: subset)
    src_meta = MetadataCatalog.get(orig_name).as_dict()
    src_meta.pop("name", None)
    MetadataCatalog.get(sub_name).set(**src_meta)
    cfg.dataloader.test.dataset.names = sub_name
    test_loader = instantiate(cfg.dataloader.test)
    # Build evaluator restricted to subset
    evaluator = COCOEvaluator(orig_name, tasks=("bbox", "segm"), distributed=False,
                              output_dir="/tmp/eva2_quant_eval")
    subset_img_ids = set(int(it["image_id"]) for it in subset)
    coco = evaluator._coco_api
    coco.imgs = {k: v for k, v in coco.imgs.items() if k in subset_img_ids}
    coco.anns = {k: v for k, v in coco.anns.items() if v["image_id"] in subset_img_ids}
    coco.imgToAnns = {k: v for k, v in coco.imgToAnns.items() if k in subset_img_ids}
    coco.catToImgs = {c: [i for i in imgs if i in subset_img_ids]
                      for c, imgs in coco.catToImgs.items()}
    return test_loader, evaluator, subset


def run_config(bits, cfg_template, test_loader, evaluator, calib_subset):
    print(f"\n{'='*60}")
    print(f"  w{bits}a{bits}  (no reparam)")
    print(f"{'='*60}")

    # Fresh model each run
    cfg = copy.deepcopy(cfg_template)
    cfg.model.backbone.net.xattn = False
    cfg.model.backbone.net.use_act_checkpoint = False
    model = instantiate(cfg.model).eval().cuda()
    DetectionCheckpointer(model).load(CKPT)

    # Collapse bias
    collapse_qkv_bias(model)

    # Quant swap
    quant_model_eva2(model, {"n_bits": bits}, {"n_bits": bits})

    # Calibrate on first N_CALIB images
    set_quant_state(model, True, True)
    print(f"[calib] running {N_CALIB} images...")
    with torch.no_grad():
        for i, item in enumerate(calib_subset[:N_CALIB]):
            import cv2
            im = cv2.imread(item["file_name"])
            img_t = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
            model([{"image": img_t.cuda(), "height": im.shape[0], "width": im.shape[1]}])
            if (i + 1) % 8 == 0:
                print(f"  calib {i+1}/{N_CALIB}")
    print("[calib] done")

    # NO reparam — go straight to eval
    print(f"[eval] running {N_EVAL} images...")
    t0 = time.time()
    results = inference_on_dataset(model, test_loader, evaluator)
    dt = time.time() - t0
    print(f"[eval] done in {dt:.1f}s")

    for task, m in results.items():
        print(f"  [{task}] AP={m.get('AP',0):.2f}  AP50={m.get('AP50',0):.2f}  AP75={m.get('AP75',0):.2f}")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    print("[0] Loading config")
    cfg_template = LazyConfig.load(CFG_PATH)

    print("[1] Building test loader")
    cfg_for_loader = copy.deepcopy(cfg_template)
    test_loader, evaluator, subset = build_subset_loader(cfg_for_loader, N_EVAL)

    OUT = Path(__file__).parent / "results" / "quant_norp"
    OUT.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for bits in [8, 6, 4]:
        # Need fresh evaluator each run (it accumulates predictions)
        _, eval_fresh, _ = build_subset_loader(copy.deepcopy(cfg_template), N_EVAL)
        results = run_config(bits, cfg_template, test_loader, eval_fresh, subset)
        all_results[f"w{bits}a{bits}"] = results

    # Save summary
    with open(OUT / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY (no reparam)")
    print(f"{'='*60}")
    print(f"{'config':<10} {'bbox AP':>8} {'bbox AP50':>9} {'segm AP':>8} {'segm AP50':>9}")
    for tag, res in all_results.items():
        b = res.get("bbox", {})
        s = res.get("segm", {})
        print(f"{tag:<10} {b.get('AP',0):>8.2f} {b.get('AP50',0):>9.2f} {s.get('AP',0):>8.2f} {s.get('AP50',0):>9.2f}")


if __name__ == "__main__":
    main()
