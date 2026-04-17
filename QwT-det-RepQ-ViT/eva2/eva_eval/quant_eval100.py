"""Quantised eval on first 100 COCO val images — EVA-02-L.
Runs w8a8, w6a6, w4a4 with reparam, then with reparam+qwt, sequentially.
"""
import os, sys, time, json, copy
from pathlib import Path
import torch

os.environ.setdefault("DETECTRON2_DATASETS", "/scratch/nbleier_owned_root/nbleier_owned1/shared_data")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVA_DET = os.path.join(_THIS_DIR, "..", "eva_det")
sys.path.insert(0, EVA_DET)
sys.path.insert(0, _THIS_DIR)

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


def build_eval(cfg_template, n_eval):
    cfg = copy.deepcopy(cfg_template)
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


def calib_forward(model, subset, n_calib):
    import cv2
    with torch.no_grad():
        for i, item in enumerate(subset[:n_calib]):
            im = cv2.imread(item["file_name"])
            img_t = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
            model([{"image": img_t.cuda(), "height": im.shape[0], "width": im.shape[1]}])
            if (i + 1) % 8 == 0:
                print(f"  calib {i+1}/{n_calib}")


def run_config(bits, cfg_template, subset, use_reparam, use_qwt):
    mode = "rp+qwt" if use_qwt else ("rp" if use_reparam else "norp")
    tag = f"w{bits}a{bits}_{mode}"
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")

    cfg = copy.deepcopy(cfg_template)
    cfg.model.backbone.net.xattn = False
    cfg.model.backbone.net.use_act_checkpoint = False
    model = instantiate(cfg.model).eval().cuda()
    DetectionCheckpointer(model).load(CKPT)

    # Collapse q_bias/v_bias into proj.bias
    if use_reparam:
        n_col = collapse_qkv_bias(model)
        print(f"[collapse] {n_col} blocks")

    # Quant swap
    quant_model_eva2(model, {"n_bits": bits}, {"n_bits": bits, "channel_wise": True})
    set_quant_state(model, True, True)

    # Calibrate
    print(f"[calib] {N_CALIB} images...")
    calib_forward(model, subset, N_CALIB)
    print("[calib] done")

    # Reparam
    if use_reparam:
        n_rp = scale_reparam_eva2(model.backbone.net)
        print(f"[reparam] {n_rp} pairs")
        # Re-calib after reparam (weight quantizers were reset)
        print(f"[re-calib] {N_CALIB} images...")
        calib_forward(model, subset, N_CALIB)
        print("[re-calib] done")

    # QwT compensation
    if use_qwt:
        qwt_loader, _, _ = build_eval(cfg_template, N_EVAL)
        t0 = time.time()
        report = generate_compensation_model_eva(
            model, qwt_loader, device=torch.device("cuda"),
            n_samples=32, start_block=0, ridge=0.0, fwd_chunk=1,
            log=lambda s: print(f"  {s}"),
        )
        r2s = [r["r2"] for r in report]
        print(f"[qwt] done in {time.time()-t0:.1f}s  "
              f"r2: min={min(r2s):.3f} mean={sum(r2s)/len(r2s):.3f} max={max(r2s):.3f}")
        set_quant_state(model, True, True)

    # Eval
    _, evaluator, _ = build_eval(cfg_template, N_EVAL)
    test_loader, _, _ = build_eval(cfg_template, N_EVAL)
    print(f"[eval] {N_EVAL} images...")
    t0 = time.time()
    results = inference_on_dataset(model, test_loader, evaluator)
    dt = time.time() - t0
    print(f"[eval] done in {dt:.1f}s")

    for task, m in results.items():
        print(f"  [{task}] AP={m.get('AP',0):.2f}  AP50={m.get('AP50',0):.2f}  AP75={m.get('AP75',0):.2f}")

    del model
    torch.cuda.empty_cache()
    return tag, results


def main():
    print("[0] Loading config")
    cfg_template = LazyConfig.load(CFG_PATH)

    print("[1] Preparing dataset")
    _, _, subset = build_eval(cfg_template, N_EVAL)

    OUT = Path(__file__).parent / "results" / "quant_rp_qwt"
    OUT.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # reparam only
    for bits in [8, 6, 4]:
        tag, res = run_config(bits, cfg_template, subset, use_reparam=True, use_qwt=False)
        all_results[tag] = res

    # reparam + qwt
    for bits in [8, 6, 4]:
        tag, res = run_config(bits, cfg_template, subset, use_reparam=True, use_qwt=True)
        all_results[tag] = res

    with open(OUT / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'config':<18} {'bbox AP':>8} {'bbox AP50':>9} {'segm AP':>8} {'segm AP50':>9}")
    for tag, res in all_results.items():
        b = res.get("bbox", {})
        s = res.get("segm", {})
        print(f"{tag:<18} {b.get('AP',0):>8.2f} {b.get('AP50',0):>9.2f} {s.get('AP',0):>8.2f} {s.get('AP50',0):>9.2f}")


if __name__ == "__main__":
    main()
