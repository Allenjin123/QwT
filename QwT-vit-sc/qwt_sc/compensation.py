"""QwT-style linear residual compensation for stochastic-computing ViTs.

Given two copies of a transformer:
  * ``model_fp`` — full-precision reference
  * ``model_sc`` — with SC matmuls patched into its attention / MLP path

we fit, per block ``i``, a small closed-form residual

    (Y_fp_i - Y_sc_i)  ~=  X_i @ W_i + b_i

on a small calibration set, then wrap each SC block as

    out_i(x) = block_sc_i(x) + x @ W_i + b_i.

This is the direct SC-analogue of the ``CompensationBlock`` in
``QwT-cls-RepQ-ViT/qwt_vit_and_deit.py`` (Fu et al., CVPR 2025), and inherits
QwT's key property: **no back-prop, closed-form LS, calibration in minutes**.

The routine is framework-agnostic about which SC kernels are used: it only
needs the two block lists (FP and SC) and a way to collect block-0 inputs.
The SC noise is only ever read through the SC block's forward, so any kernel
(XNOR/Sobol bitstream, calibrated Gaussian surrogate, etc.) works.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn


class CompensationBlock(nn.Module):
    """Residual linear correction wrapping an inner block.

    ``out = block(x) + x @ W + b``, with ``W`` of shape ``(D_in, D_out)``.
    The correction is applied as an ``nn.Linear`` so that trailing-dim inputs
    ``(B, N, D_in)`` are handled naturally across token dimensions.

    Args:
        block: the inner block (quantized / SC / anything).
        W: ``(D_in, D_out)`` tensor from the LS fit.
        b: ``(D_out,)`` tensor.
        r2: coefficient of determination from the fit (diagnostic).
        enabled: if ``False``, the wrapper is a no-op — useful when the fit
            was degenerate (very low r^2) and you'd rather leave the block
            uncompensated.
    """

    def __init__(self, block: nn.Module, W: torch.Tensor, b: torch.Tensor,
                 r2: float = 0.0, enabled: bool = True):
        super().__init__()
        self.block = block
        D_in, D_out = W.shape
        self.comp = nn.Linear(D_in, D_out, bias=True)
        with torch.no_grad():
            self.comp.weight.copy_(W.t().contiguous())
            self.comp.bias.copy_(b)
        self.r2 = float(r2)
        self.enabled = bool(enabled)

    def forward(self, x, *args, **kwargs):
        out = self.block(x, *args, **kwargs)
        if not self.enabled:
            return out
        return out + self.comp(x)


def closed_form_ridge(X: torch.Tensor, Y: torch.Tensor,
                      ridge: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve ``min_{W,b} ||Y - (X W + b)||^2 + ridge * ||W||^2`` in closed form.

    Stacks an all-ones column to fit the bias, and ridges only the slope
    (standard convention). Uses ``torch.linalg.solve`` on the ``(D+1, D+1)``
    Gram matrix — exact, no backprop.

    Args:
        X: ``(N, D_in)`` feature matrix.
        Y: ``(N, D_out)`` target matrix.
        ridge: L2 regularization strength on ``W`` (not on ``b``).

    Returns:
        ``W`` ``(D_in, D_out)``, ``b`` ``(D_out,)``, ``r2`` (scalar, unregularized).
    """
    N, D = X.shape
    ones = torch.ones(N, 1, device=X.device, dtype=X.dtype)
    X_aug = torch.cat([X, ones], dim=-1)
    G = X_aug.t() @ X_aug
    reg = torch.zeros_like(G)
    reg[:D, :D] = ridge * torch.eye(D, device=G.device, dtype=G.dtype)
    rhs = X_aug.t() @ Y
    sol = torch.linalg.solve(G + reg, rhs)
    W = sol[:-1]
    b = sol[-1]
    Y_mean = Y.mean(dim=0, keepdim=True)
    ss_tot = ((Y - Y_mean) ** 2).sum()
    ss_res = ((Y - (X @ W + b)) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot).item() if ss_tot.item() > 0 else 0.0
    return W, b, r2


def _forward_batched(block: nn.Module, X: torch.Tensor, device: torch.device,
                     chunk: int) -> torch.Tensor:
    """Run ``block`` on CPU-resident ``X`` batch-by-batch, return CPU outputs."""
    outs = []
    for s in range(0, X.size(0), chunk):
        xb = X[s:s + chunk].to(device, non_blocking=True)
        y = block(xb)
        outs.append(y.detach().float().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def calibrate_qwt(
    model_fp: nn.Module,
    model_sc: nn.Module,
    blocks_fp: Sequence[nn.Module],
    blocks_sc_container,  # supports __getitem__ and __setitem__ (e.g. nn.ModuleList)
    calib_loader,
    device: torch.device,
    n_calib: int,
    ridge: float = 1e-2,
    start_block: int = 0,
    fwd_chunk: int = 32,
    avg_sc_draws: int = 1,
    log_fn: Callable[[str], None] = print,
) -> List[dict]:
    """Sequentially fit and install ``CompensationBlock`` wrappers on ``model_sc``.

    For each block ``i``, using the *same* input as the SC path sees (important
    — errors compound), this function:

    1. Computes the FP target ``Y_fp_i = blocks_fp[i](X_i)``.
    2. Computes the SC output ``Y_sc_i = blocks_sc[i](X_i)`` — optionally
       averaging ``avg_sc_draws`` independent realizations for a lower-variance
       estimate of ``E[Y_sc | X]`` when the SC kernel is stochastic.
    3. Solves ``(Y_fp - Y_sc) = X W + b`` via :func:`closed_form_ridge`.
    4. Wraps ``blocks_sc[i]`` with :class:`CompensationBlock(W, b)` and
       *propagates X through the compensated block* to form ``X_{i+1}``. This
       sequential (Gauss–Seidel) update matches QwT's original scheme and is
       crucial: each block is fit on the distribution it will actually see at
       inference, after upstream compensation.

    The function mutates ``blocks_sc_container[i]`` in place. It does not touch
    ``blocks_fp``.

    Args:
        model_fp: FP reference model; used only to gather block-0 inputs via
            a forward hook on its first block (``blocks_fp[0]``). You may pass
            ``model_sc`` here instead if you want to use SC-path inputs.
        model_sc: The SC-patched model we are compensating.
        blocks_fp: List/sequence of FP blocks (same length as ``blocks_sc_container``).
        blocks_sc_container: Container of SC blocks supporting item assignment
            (typically ``model_sc.backbone.blocks`` or ``model_sc.blocks``).
        calib_loader: yields ``(images, labels)`` or ``(images, _)``; only the
            images are consumed.
        device: torch device to run forwards on.
        n_calib: number of calibration samples to gather.
        ridge: L2 reg on the residual weight.
        start_block: blocks with index ``< start_block`` get ``enabled=False``
            (pass-through) after wrapping — useful if early blocks have noisy
            fits that hurt downstream.
        fwd_chunk: number of calib samples per block forward chunk.
        avg_sc_draws: if > 1, average this many independent SC passes through
            each block during calibration to reduce target variance.
        log_fn: logging callback; defaults to ``print``.

    Returns:
        A list of per-block diagnostic dicts with keys
        ``{block, r2, enabled, rmse_before, rmse_after, dt_s}``.
    """
    assert len(blocks_fp) == len(blocks_sc_container), (
        f"fp blocks ({len(blocks_fp)}) vs sc blocks "
        f"({len(blocks_sc_container)}) length mismatch"
    )
    n_blocks = len(blocks_fp)

    # Collect block-0 inputs via a forward pre-hook on model_sc's first block.
    first_block = blocks_sc_container[0]
    captured: List[torch.Tensor] = []

    def hook(_m, args):
        captured.append(args[0].detach().float().cpu())

    log_fn(f"[calib] collecting first-block inputs from model_sc")
    handle = first_block.register_forward_pre_hook(hook)
    seen = 0
    try:
        for batch in calib_loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.to(device, non_blocking=True)
            model_sc(imgs)
            seen += imgs.size(0)
            if seen >= n_calib:
                break
    finally:
        handle.remove()
    X_cur = torch.cat(captured, dim=0)[:n_calib]
    log_fn(f"[calib] X_0 shape={tuple(X_cur.shape)}  dtype={X_cur.dtype}")

    report: List[dict] = []
    for i in range(n_blocks):
        t0 = time.time()
        blk_fp = blocks_fp[i]
        blk_sc = blocks_sc_container[i]

        Y_fp = _forward_batched(blk_fp, X_cur, device, fwd_chunk)
        Y_sc = None
        for _ in range(avg_sc_draws):
            yd = _forward_batched(blk_sc, X_cur, device, fwd_chunk)
            Y_sc = yd if Y_sc is None else Y_sc + yd
        Y_sc = Y_sc / avg_sc_draws

        X_flat = X_cur.reshape(-1, X_cur.size(-1))
        R_flat = (Y_fp - Y_sc).reshape(-1, Y_fp.size(-1))
        Xg = X_flat.to(device)
        Rg = R_flat.to(device)
        W, b, r2 = closed_form_ridge(Xg, Rg, ridge=ridge)

        raw_rmse = (Y_fp - Y_sc).pow(2).mean().sqrt().item()
        after_rmse = (Rg - (Xg @ W + b)).pow(2).mean().sqrt().item()

        enabled = (i >= start_block) and (r2 > 0.0)
        new_block = CompensationBlock(
            block=blk_sc, W=W.detach().cpu(), b=b.detach().cpu(),
            r2=r2, enabled=enabled,
        ).to(device)
        blocks_sc_container[i] = new_block

        X_cur = _forward_batched(new_block, X_cur, device, fwd_chunk)

        dt = time.time() - t0
        log_fn(
            f"[calib] block {i:2d}  r2={r2:+.4f}  "
            f"rmse {raw_rmse:.4e} -> {after_rmse:.4e}  "
            f"enabled={enabled}  ({dt:.1f}s)"
        )
        report.append({
            "block": i, "r2": r2, "enabled": enabled,
            "rmse_before": raw_rmse, "rmse_after": after_rmse,
            "dt_s": dt,
        })

        del Xg, Rg, W, b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return report
