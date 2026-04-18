"""qwt_sc — QwT-style closed-form residual compensation for stochastic-
computing (SC) ViTs.

Drop-in port of ``QwT-cls-RepQ-ViT/qwt_vit_and_deit.py``'s ``CompensationBlock``
idea to the SC setting (bitstream/Sobol matmuls instead of INT8 PTQ). The
compensation is per-block ``out = block_sc(x) + x W + b`` where ``W, b`` are
fit in one shot by closed-form ridge least-squares on a small calibration set,
using an FP reference pass of the same block as the regression target. No
back-prop, no fine-tuning.

Two calibration entry points:

* ``calibrate_qwt`` — full-model API, takes an FP model, SC model, and a
  calibration DataLoader; wraps each block in ``CompensationBlock`` in place.
  Use for image-classifier-style models where the block list is the comp
  installation point (e.g. ``model_sc.backbone.blocks``).

* ``calibrate_block_residuals`` — block-list API, takes two parallel
  ``[(attn, ff), ...]`` lists and an explicit calibration tensor; returns
  per-block ``(W, b)`` without wrapping. Use for models whose comp modules
  live in a separate ``nn.ModuleList`` (e.g. the Dino-WM ``SCViTPredictor``
  with ``transformer.comp_blocks``).

See ``../README.md`` for the formulation, usage, and results on DINOv2
ViT-L/14 + the ImageNet-1k linear head.
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
