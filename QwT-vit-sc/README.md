# QwT for stochastic-computing ViTs (`QwT-vit-sc`)

A port of [QwT](https://arxiv.org/abs/2411.13918)'s closed-form `CompensationBlock`
from integer-quantized ViTs (the `QwT-cls-RepQ-ViT/` baseline in this repo) to
**stochastic-computing (SC) ViTs** — where block-level matmuls are swapped for
bitstream / Sobol-SC kernels and the accuracy loss comes from random SC noise
rather than deterministic quantization error.

The compensation recipe is the same one-line formula as QwT:

```
out_i(x)  =  block_sc_i(x)  +  x @ W_i + b_i
```

with `W_i, b_i` fit in closed form by ridge LS on a small calibration set,
using an FP reference pass of the same block as the regression target. No
back-prop, no fine-tuning.

## What this module provides

A tiny framework-agnostic library (`qwt_sc/`) exposing:

| symbol | purpose |
|---|---|
| `CompensationBlock(block, W, b)` | `nn.Module` wrapping an SC block with a residual `x·W + b`. |
| `closed_form_ridge(X, Y, ridge)` | One-shot `(Xᵀ X + λ I)⁻¹ Xᵀ Y` solve with bias. Returns `W, b, r²`. |
| `calibrate_qwt(model_fp, model_sc, blocks_fp, blocks_sc_container, loader, ...)` | Sequentially fit + install `CompensationBlock`s on every block of `model_sc`, block-by-block, using the Gauss-Seidel propagation scheme that QwT also uses. |

It does **not** bundle any SC kernel or any specific model — you bring your
own. A complete end-to-end driver for DINOv2 ViT-L/14 + the
[`vit_sc`](https://github.com/Polarisyjr/vit_sc) SC kernels lives at
`vit_sc/experiments/qwt_sc_compensation.py`.

## Usage

```python
import torch
from qwt_sc import calibrate_qwt

# 1. Load two copies with identical weights.
model_fp = load_my_vit().to(device).eval()
model_sc = load_my_vit().to(device).eval()

# 2. Patch model_sc with your SC kernels. After this call every block of
#    model_sc runs its matmuls in SC; model_fp is untouched.
patch_with_sc(model_sc, sc_prec=8, ...)

# 3. Grab parallel block handles. Any indexable container works.
blocks_fp = list(model_fp.backbone.blocks)
blocks_sc_container = model_sc.backbone.blocks   # nn.ModuleList / Sequential

# 4. Fit + install compensation blocks in place.
report = calibrate_qwt(
    model_fp=model_fp,
    model_sc=model_sc,
    blocks_fp=blocks_fp,
    blocks_sc_container=blocks_sc_container,
    calib_loader=calib_loader,    # any iterable yielding (images, _)
    device=device,
    n_calib=256,
    ridge=1e-2,
    fwd_chunk=32,
    avg_sc_draws=1,               # >1 to average SC realizations during calib
    # ─── optional: SC-kernel compensator ────────────────────────────
    # comp_factory=...,           # see "SC compensator" section below
    # comp_factory_variants=...,  # per-block variant selection
    # comp_refit_iters=0,         # noise-aware refit iterations
)

# 5. model_sc is now a QwT-compensated SC model. Use as-is.
acc = evaluate(model_sc, val_loader, device)
```

## SC compensator (run the residual itself in SC)

By default `calibrate_qwt` installs an FP `nn.Linear` per block as the
residual — fast on a GPU but requires an FP unit on an SC accelerator. To
remove that dependency, supply a `comp_factory(W, b) → nn.Module` that
returns an SC-kernel module instead. The closed-form ridge fit is unchanged;
only the inference-time matmul flips to SC.

```python
from qwt_sc import calibrate_qwt

# Caller supplies the SC matmul (e.g., vit_sc's SCLinear).
def sc_comp_factory(W, b):
    D_in, D_out = W.shape
    layer = nn.Linear(D_in, D_out, bias=True)
    with torch.no_grad():
        layer.weight.copy_(W.t().contiguous())
        layer.bias.copy_(b)
    return SCLinear(layer, sc_prec=8, mode="bipolar")

calibrate_qwt(
    ..., comp_factory=sc_comp_factory,
)
```

Two extra knobs:

- `comp_refit_iters=K` — *noise-aware refit*: at each block, after fitting
  `W₀` via standard ridge, measure the SC comp's actual noise
  `δ(X; W₀, r_c) := c_sc(X; W₀) − (X W₀ + b₀)` on the calibration set, then
  refit on `R − δ`. Linearizes `c_sc` around `W₀`. Useful in principle; in
  practice can overshoot on early/small-residual blocks where the comp's SC
  noise floor exceeds the residual it's trying to fit.

- `comp_factory_variants=[(name, factory), ...]` — per-block variant
  selection. The lib runs each candidate factory on the calibration set,
  measures `‖R − c_sc(X; W)‖`, and installs the lower-residual one.
  Records the chosen name in the per-block report (= 1 config bit per block
  in HW when `len(variants) == 2`). Used with two Sobol scrambling configs
  (default vs antithetic-K) — empirically close calls in our kernel because
  value-complement antithetic gives only ~−0.15 noise correlation here.

### Results: SC-comp vs FP-comp (same calibration, same eval)

DINOv2 ViT-L/14 + ImageNet-1k, N=500, sc_prec=8, ridge=1e-2:

| sc_config | FP-comp | SC-comp (naive) | Δ |
|---|---:|---:|---:|
| `qk_only`     | 0.834 | 0.810 | **−2.4 pt** (block uses D=64 per-head SNG; comp at D=1024 → independent noise) |
| `skip_worst50`| 0.836 | **0.840** | **+0.4 pt** (block has D=1024 SC ops; pool collides with comp via cache → free control-variate cancellation) |

The take-away mirrors a control-variate argument: SC-comp matches or beats
FP-comp wherever the comp's Sobol pool overlaps a block SC op at the same
dim. Where it doesn't, SC-comp's noise is informationally independent of
the block residual and stacks with it. **Open work**: a head-aligned SC
compensator that uses the block's per-head QK SNG bank inside the comp
matmul, restoring the pool overlap on `qk_only`-style configs.

That's it — no gradient, no hyperparameter search, one pass. Calibration on
256 images for a 24-block ViT-L/14 takes roughly 3–5 min on a single consumer
GPU.

## Results on DINOv2 ViT-L/14 + ImageNet-1k

Run on `vit_sc` (RTX 4080, N=500 val images, seed=0, `sc_prec=8`, 256 calib
images, `ridge=1e-2`, single SC draw during calibration):

| SC config | SC ops | Raw SC top-1 | **+ QwT comp** | Δ | Gap-to-FP closed |
|---|---:|---:|---:|---:|---:|
| FP baseline           | 0/120   | **0.846** | — | — | — |
| `full_attn` (qk+av+qkv_proj+out_proj) | 72/120 | 0.696 | **0.786** | +9.0 pt | 60% |
| `qk_av`               | 48/120  | 0.718 | **0.822** | +10.4 pt | 81% |
| `qk_only`             | 24/120  | 0.736 | **0.834** | +9.8 pt  | 89% |
| `skip_worst50` (fine-grained) | 70/120 | 0.748 | **0.836** | +8.8 pt  | 90% |

Take-aways:
- **+8.8 to +10.4 pt top-1 across every tested SC config**, uniformly.
- **Stacks with sensitivity-aware SC scheduling**: `skip_worst50 + QwT` lands
  at 83.6% (within 1.0 pt of FP) at 70/120 SC ops.
- **Pareto shift**: `qk_only + QwT` (83.4% with only 24 SC ops) now strictly
  dominates `skip_worst50` *without* compensation (74.8% with 70 SC ops).
- Recovers more where the baseline is higher (89–90% of gap closed for
  `qk_only` / `skip_worst50` vs 60% for the noisiest `full_attn`), consistent
  with the picture that a *linear* residual mainly fixes the mean-shift part
  of the SC residual — variance-dominated configs have less linear structure
  to grab.

## Overhead

Per block (ViT-L/14, `D=1024`, `N=257` tokens):
- **Params added**: `D² + D = 1,049,600` (≈ 1.05 M per block → **+25 M total**, ~8.4% of the ViT-L backbone).
- **FLOPs added per image**: 2 · N · D² ≈ **0.54 GFLOPs** per block → ~13 GFLOPs / image → **+21% over FP inference FLOPs**.
- Calibration memory: transient ~1.5 GB on GPU; no per-sample state at inference.
- Wall-clock at inference: lost in the SC bottleneck (SC: 175s vs SC+comp: 176s on N=500).

Mirrors QwT's own fp16-compensation-over-INT8-backbone story: a light FP
linear per block recovers a large fraction of the accuracy lost in the
heavier quantized path.

## What's *not* compensable (and why)

A static linear `W, b` fits the best linear approximation of
`E[Y_fp − Y_sc | X]`. It captures:
- per-dim biases,
- linear mean-shift along `x`,
- cross-dim correlated drift.

What it can't chase:
- the per-sample, Sobol-realization-dependent variance around that mean.

In the `vit_sc` kernels the Sobol state is deterministic given a seed, so the
"random" part is really systematic, input-dependent residual. With nonlinear
features of `x` (e.g. `x ⊙ x`), Sobol-pool-aware features, or antithetic SC
draws, the remaining gap can in principle be attacked further — but linear
already gets 60–90% of the way on all configs we tested.

## Reproducing the results

1. Clone `vit_sc` (which pins this module as a submodule at
   `third_party/QwT-SC`) and set up its conda env as its README describes.
2. `cd vit_sc && python experiments/qwt_sc_compensation.py --sc_config skip_worst50 --n_calib 256 --n_eval 500 --out_json results/my_run.json`
3. Results land in `results/qwt_sc_<config>_n500.json`.

## Citation

If you use this port, please also cite the original QwT paper:

```bibtex
@InProceedings{Fu_2025_CVPR,
    author    = {Fu, Minghao and Yu, Hao and Shao, Jie and Zhou, Junjie and Zhu, Ke and Wu, Jianxin},
    title     = {Quantization without Tears},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {4462-4472}
}
```
