"""qwt_sc — QwT-style closed-form residual compensation for stochastic-
computing (SC) ViTs.

Drop-in port of ``QwT-cls-RepQ-ViT/qwt_vit_and_deit.py``'s ``CompensationBlock``
idea to the SC setting (bitstream/Sobol matmuls instead of INT8 PTQ). The
compensation is per-block ``out = block_sc(x) + x W + b`` where ``W, b`` are
fit in one shot by closed-form ridge least-squares on a small calibration set,
using an FP reference pass of the same block as the regression target. No
back-prop, no fine-tuning.

See ``../README.md`` for the formulation, usage, and results on DINOv2
ViT-L/14 + the ImageNet-1k linear head.
"""
from .compensation import (
    CompensationBlock,
    closed_form_ridge,
    calibrate_qwt,
)

__all__ = [
    "CompensationBlock",
    "closed_form_ridge",
    "calibrate_qwt",
]
