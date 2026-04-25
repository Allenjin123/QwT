"""qwt_sc — QwT-style closed-form residual compensation for stochastic-
computing (SC) ViTs.

Drop-in port of ``QwT-cls-RepQ-ViT/qwt_vit_and_deit.py``'s ``CompensationBlock``
idea to the SC setting (bitstream/Sobol matmuls instead of INT8 PTQ). The
compensation is per-block ``out = block_sc(x) + x W + b`` where ``W, b`` are
fit in one shot by closed-form ridge least-squares on a small calibration
set, using an FP reference pass of the same block as the regression target.
No back-prop, no fine-tuning.

Calibration entry points:

* ``calibrate_qwt`` — full-model API with the **cross-seed cosine gate**.
  Takes an FP model, SC model, and *two* disjoint calibration DataLoaders;
  fits ``W_A, W_B`` on each batch and installs ``W̄ = (W_A + W_B)/2`` iff
  ``cos(W_A, W_B) > τ``. This is the single-recipe gate — one ``τ`` works
  across all five of the cls sweep regimes (p7_u, p8_u, avg192_u, p7_mp,
  avg192_mp). See ``compensation.py`` module docstring for the physical
  derivation and pilot evidence.

* ``calibrate_block_residuals`` — block-list API for models whose comp
  modules live in a separate ``nn.ModuleList`` (e.g. the Dino-WM
  ``SCViTPredictor`` with ``transformer.comp_blocks``). Takes parallel
  ``[(attn, ff), ...]`` lists and an explicit calibration tensor; returns
  per-block ``(W, b)`` without wrapping.

See ``../README.md`` for formulation, usage, and results on DINOv2 ViT-L/14
+ the ImageNet-1k linear head.
"""
from .compensation import (
    CompensationBlock,
    closed_form_ridge,
    calibrate_qwt,
)
from .block_list_comp import (
    BlockResidualFn,
    calibrate_block_residuals,
    save_comp_weights,
    load_comp_weights,
)

__all__ = [
    "CompensationBlock",
    "closed_form_ridge",
    "calibrate_qwt",
    "BlockResidualFn",
    "calibrate_block_residuals",
    "save_comp_weights",
    "load_comp_weights",
]
