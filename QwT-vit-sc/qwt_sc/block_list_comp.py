"""Block-list flavour of QwT-SC compensation for ViT-style predictors.

Distinct from ``calibrate_qwt`` (which takes a full model + DataLoader and
wraps each block in a ``CompensationBlock``): this API takes two parallel
lists of ``(attn, ff)`` pairs (an FP reference set and a matched SC set) and
an explicit calibration tensor ``calib_X``, and returns the per-block
residual ``(W, b)`` plus diagnostics. The caller decides how to install the
residuals — useful when the host model keeps comp modules in a separate
``nn.ModuleList`` rather than wrapping each transformer block in place
(e.g. ``SCViTPredictor.transformer.comp_blocks``).

Note on stochasticity / ``avg_sc_draws``:
    The SC kernels in this repo can be *deterministic given input* when the
    Sobol config has fixed seeds (see ``make_sobol_simple_config``). In that
    regime, averaging several calls with the same X does not reduce target
    variance — it returns the same value. Averaging is only useful with
    stochastic RNGs (e.g. LFSRs with random seeds drawn per-call, or the
    Gaussian surrogate in ``noise_matmul``). For deterministic Sobol the
    residual ``R = Y_fp − Y_sc`` is a (non-linear) function of X; the
    appropriate knobs to prevent over-fitting are ``calib_X`` size and
    ``ridge``, not ``avg_sc_draws``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence

import torch
from torch import nn

from .compensation import closed_form_ridge


class BlockResidualFn(nn.Module):
    """Wraps an ``(attn, ff)`` pair so it forwards as one residual block:

        x = attn(x) + x
        x = ff(x) + x
    """
    def __init__(self, attn: nn.Module, ff: nn.Module):
        super().__init__()
        self.attn = attn
        self.ff = ff

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


@torch.no_grad()
def calibrate_block_residuals(
    fp_layers: Sequence,
    sc_layers: Sequence,
    calib_X: torch.Tensor,
    ridge: float = 1e-2,
    avg_sc_draws: int = 1,
    start_block: int = 0,
    log_fn: Callable[[str], None] = print,
) -> List[dict]:
    """Sequential (Gauss-Seidel) ridge LS fit of per-block residuals.

    For each block i:
        Y_fp  = block_fp(X_cur)               # FP target
        Y_sc  = mean_{avg_sc_draws}(block_sc(X_cur))
        R     = Y_fp - Y_sc
        [W,b] = closed_form_ridge(X_cur, R)
        X_cur = Y_sc + (X_cur W + b)          # propagate compensated out

    Args:
        fp_layers: sequence where ``fp_layers[i][0]`` is the FP attn module
            and ``fp_layers[i][1]`` is the FP ff module.
        sc_layers: same structure as ``fp_layers`` but SC-patched.
        calib_X: ``(N, T, D)`` tensor that matches the transformer's input
            distribution (post-pos-embedding, post-dropout).
        ridge: L2 reg on the slope W (no reg on bias).
        avg_sc_draws: average this many SC forward passes — only meaningful
            if the SC kernel is stochastic (see module docstring).
        start_block: blocks ``< start_block`` are marked ``enabled=False``.
        log_fn: logging callback.

    Returns:
        list of per-block dicts:
            {"block", "W", "b", "r2", "enabled", "rmse_before", "rmse_after"}
    """
    depth = len(sc_layers)
    assert len(fp_layers) == depth, (
        f"fp layers {len(fp_layers)} != sc layers {depth}"
    )

    device = next(sc_layers[0][0].parameters()).device
    X_cur = calib_X.to(device).float()
    log_fn(f"[comp-calib] X_0 shape={tuple(X_cur.shape)} dtype={X_cur.dtype}")

    report: List[dict] = []
    for i in range(depth):
        fp_attn, fp_ff = fp_layers[i][0], fp_layers[i][1]
        sc_attn, sc_ff = sc_layers[i][0], sc_layers[i][1]
        fp_block = BlockResidualFn(fp_attn, fp_ff).to(device).eval()
        sc_block = BlockResidualFn(sc_attn, sc_ff).to(device).eval()

        Y_fp = fp_block(X_cur).float()
        Y_sc = sc_block(X_cur).float()
        for _ in range(avg_sc_draws - 1):
            Y_sc = Y_sc + sc_block(X_cur).float()
        Y_sc = Y_sc / float(avg_sc_draws)

        D = X_cur.shape[-1]
        X_flat = X_cur.reshape(-1, D)
        R_flat = (Y_fp - Y_sc).reshape(-1, D)

        W, b, r2 = closed_form_ridge(X_flat, R_flat, ridge=ridge)
        rmse_before = (Y_fp - Y_sc).pow(2).mean().sqrt().item()
        rmse_after = (R_flat - (X_flat @ W + b)).pow(2).mean().sqrt().item()

        enabled = (i >= start_block) and (r2 > 0.0)
        log_fn(f"[comp-calib] block {i}  r2={r2:+.4f}  "
               f"rmse {rmse_before:.4e} -> {rmse_after:.4e}  "
               f"||W||={W.norm().item():.2f} ||b||={b.norm().item():.2f}  "
               f"enabled={enabled}")
        report.append({
            "block": int(i),
            "W": W.detach().cpu(),
            "b": b.detach().cpu(),
            "r2": float(r2),
            "enabled": bool(enabled),
            "rmse_before": float(rmse_before),
            "rmse_after": float(rmse_after),
        })

        if enabled:
            comp_out = X_flat @ W + b
            X_cur = (Y_sc + comp_out.reshape(*Y_sc.shape)).detach()
        else:
            X_cur = Y_sc.detach()

    return report


def save_comp_weights(report: List[dict], depth: int, dim: int,
                      ridge: float, path: str) -> None:
    """Write the per-block ``(W, b)`` report and metadata to a torch file."""
    payload = {
        "depth": int(depth),
        "dim": int(dim),
        "ridge": float(ridge),
        "blocks": [
            {"W": r["W"], "b": r["b"], "r2": r["r2"], "enabled": r["enabled"]}
            for r in report
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_comp_weights(path: str) -> dict:
    """Inverse of ``save_comp_weights``."""
    return torch.load(path, map_location="cpu", weights_only=False)
