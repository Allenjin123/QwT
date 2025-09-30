# QwT‑mm‑RepQ‑ViT

Quantization and evaluation toolkit for **CLIP‑style multi‑modal models** (image + text) that combines **RepQ‑ViT** (scale reparameterization PTQ) with **QwT** (lightweight linear compensation). It supports:

- **FP32 zero‑shot evaluation** on ImageNet‑1K.
- **Image‑only PTQ** (visual encoder), suitable for CLIP‑based classifiers.
- **All‑modules PTQ** (visual + text encoders) for retrieval/zero‑shot scenarios.
- Optional **QwT compensation** to further close the FP32→INT gap with tiny overhead.

---

## 1. Recommended Dependencies
  - Timm v0.4.12
  - PyTorch ≥ 2.0
  - webdataset v0.2.100
---

## 2. Data Preparation

### 2.1 ImageNet‑1K (val set) for zero‑shot
Organize the **ImageNet validation** directory in the standard layout:
```
/path/to/imagenet/val/
  n01440764/ILSVRC2012_val_00000293.JPEG
  n01440764/...
  ...
```

### 3.2 Full calibration set (WebDataset)

To **reproduce our reported results**, please use the **full CC3M WebDataset** for calibration (all shards). Small subsets (e.g., 512 images) lead to **unstable or optimistic variance**, especially at low bit-widths, and cannot guarantee the paper’s numbers.

**Example config (brace expansion over all shards):**
```angular2html
--train-data "/path/to/cc3m-train-{0000..0575}.tar"
--dataset-type webdataset
```

**Notes**
- Use the **entire CC3M shard list** to obtain stable calibration statistics.
- For **quick debugging only**, you may start with a few shards (e.g., `{0000..0003}`) to verify the pipeline, but **do not** expect final accuracy from such subsets.
- For **image-only PTQ**, the forward pass does not consume captions; however, keeping real captions in shards preserves the loader format and makes it easy to switch to all-modules PTQ later.


---

## 3. Quick Start

### 3.1 FP32 zero‑shot evaluation (no quantization, running in fp32)
```bash
python main.py \
  --choice fp32_eval \
  --model "ViT-B/32" \
  --imagenet-val /path/to/imagenet/val \
  --batch-size 128
```
* This builds CLIP zero‑shot weights from prompt templates and evaluates Top‑1/Top‑5 on ImageNet‑val.
* `--wq_parmas` and `--aq_params` specify the quantization bitwise.

### 3.2 Image‑only PTQ (visual encoder)
Quantize only the visual backbone; useful for **classification** when the text path is not used at inference.
```bash
python main.py \
  --choice image_only \
  --model "ViT-B/32" \
  --imagenet-val /path/to/imagenet/val \
  --train-data "/path/to/cc3m-train-{0000..0575}.tar" \
  --dataset-type webdataset \
  --batch-size 128 \
  --iter 4 \
  --wq_params 6 --aq_params 6 \
  (--qwerty)
```
* Add `--qwerty` to enable **QwT** compensation (recommended for low‑bit).   
* `--wq_parmas` and `--aq_params` specify the quantization bitwise.

### 3.3 All‑modules PTQ (visual + text)
Quantize **both** encoders for **retrieval / zero‑shot** usage.
```bash
python main.py \
  --choice all_quant \
  --model "ViT-B/32" \
  --imagenet-val /path/to/imagenet/val \
  --train-data "/path/to/cc3m-train-{0000..0575}.tar" \
  --dataset-type webdataset \
  --batch-size 128 \
  --iter 4 \
  --wq_params 6 --aq_params 6 \
  (--qwerty)
```
Add `--qwerty` to enable **QwT** compensation (recommended for low‑bit).

> See `script/run.sh` for a minimal example. You can switch models to `"ViT-B/16"`, `"ViT-L/14"`, etc.

---

## 4. Quantization results for zero-shot classification on ImageNet

| Quant Setup   | Method             | #Bits | Top-1 |
|---|---|---:|---:|
| Vision        | Full-precision     | 32/32 | 63.4 |
| Vision        | RepQ-ViT      | 6/6   | 59.2 |
| Vision        | **RepQ-ViT + QwT** | 6/6   | **60.3** |
| Vision        | RepQ-ViT      | 8/8   | 62.9 |
| Vision        | **RepQ-ViT + QwT** | 8/8   | **63.0** |
| Vision & Text | Full-precision     | 32/32 | 63.4 |
| Vision & Text | RepQ-ViT      | 6/6   | 29.8 |
| Vision & Text | **RepQ-ViT + QwT** | 6/6   | **43.5** |
| Vision & Text | RepQ-ViT      | 8/8   | 38.7 |
| Vision & Text | **RepQ-ViT + QwT** | 8/8   | **54.6** |




## 5. Citation

We would greatly appreciate it if you could cite our paper if you find our implementation helpful in your work.

```bash
@InProceedings{Fu_2025_CVPR,
    author    = {Fu, Minghao and Yu, Hao and Shao, Jie and Zhou, Junjie and Zhu, Ke and Wu, Jianxin},
    title     = {Quantization without Tears},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {4462-4472}
}