"""QwT residual compensation for EVA-01 ViT, ported from
QwT-det-RepQ-ViT/tools/test.py (Swin-based original).

Key simplifications vs Swin port:
  * EVA ViT is monolithic: single `blocks` ModuleList, no per-stage downsample,
    no window attention mask argument. Block forward is `block(x)` with 4D
    input (B, H, W, C).
  * Single-GPU LS fit (no DDP gather).

Public surface:
  class CompensationBlock(W, b, r2_score, block, linear_init, block_id)
  generate_compensation_model_eva(q_vit, calib_loader, device, n_samples,
                                   start_block=0, ridge=0.0)
"""
from __future__ import annotations
import time
from typing import List

import torch
import torch.nn as nn

from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul


class CompensationBlock(nn.Module):
    """Wrap an EVA ViT Block with an additive residual `x @ W + b`."""

    def __init__(self, W: torch.Tensor, b: torch.Tensor, r2_score: float,
                 block: nn.Module, linear_init: bool = True, block_id: int = 0):
        super().__init__()
        self.block = block
        self.lora_weight = nn.Parameter(torch.zeros(W.size(0), W.size(1)))
        self.lora_bias   = nn.Parameter(torch.zeros(W.size(1)))
        self.r2_score = float(r2_score)
        if linear_init and r2_score > 0:
            self.lora_weight.data.copy_(W)
            self.lora_bias.data.copy_(b)
            self._init = "linear"
        else:
            nn.init.zeros_(self.lora_weight)
            nn.init.zeros_(self.lora_bias)
            self._init = "zero"
        self.block_id = block_id

    def forward(self, x):
        out = self.block(x)
        return out + x @ self.lora_weight + self.lora_bias


def enable_quant(submodel):
    for m in submodel.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(True, True)


def disable_quant(submodel):
    for m in submodel.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(False, False)


def _linear_regression(X: torch.Tensor, Y: torch.Tensor, ridge: float = 0.0):
    """Closed-form LS with intercept. X (N, D_in), Y (N, D_out).
    Returns W (D_in, D_out), b (D_out,), r2_score (scalar)."""
    X = X.reshape(-1, X.size(-1))
    Y = Y.reshape(-1, Y.size(-1))
    ones = torch.ones(X.size(0), 1, device=X.device, dtype=X.dtype)
    X_aug = torch.cat([X, ones], dim=-1)
    G = X_aug.t() @ X_aug
    if ridge > 0:
        reg = torch.zeros_like(G)
        D = X.size(-1)
        reg[:D, :D] = ridge * torch.eye(D, device=G.device, dtype=G.dtype)
        G = G + reg
    rhs = X_aug.t() @ Y
    W_overall = torch.linalg.solve(G, rhs)
    W = W_overall[:-1, :]
    b = W_overall[-1, :]

    Y_pred = X @ W + b
    ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum()
    ss_res = ((Y - Y_pred) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
    abs_err = (Y - Y_pred).abs().mean().item()
    return W, b, r2, abs_err


@torch.no_grad()
def generate_compensation_model_eva(
    model,                  # full D2 model; we hook model.backbone.net.blocks[0]
    calib_loader,           # yields D2-style list-of-dicts batches
    device: torch.device,
    n_samples: int,
    start_block: int = 0,
    ridge: float = 0.0,
    fwd_chunk: int = 1,
    log=print,
) -> List[dict]:
    """Walk through each ViT block, fit `(Y_fp - Y_quant) = X W + b` per block,
    wrap it with `CompensationBlock`. Propagates X through the compensated
    (quant ON + comp ON) block to form the next block's input, matching QwT's
    Gauss-Seidel scheme."""
    vit = model.backbone.net
    blocks = vit.blocks

    # ---- 1. Collect first-block inputs from the quantised model ----
    captured: List[torch.Tensor] = []

    def hook(_m, args):
        captured.append(args[0].detach().float().cpu())

    h = blocks[0].register_forward_pre_hook(hook)
    log(f"[qwt] collecting first-block inputs: need ~{n_samples} images")
    seen = 0
    try:
        enable_quant(model)
        for batch in calib_loader:
            model(batch)
            seen += len(batch)
            if seen >= n_samples:
                break
    finally:
        h.remove()

    X_cur = torch.cat(captured, dim=0)[:n_samples]
    log(f"[qwt] X_0 shape={tuple(X_cur.shape)}  dtype={X_cur.dtype}")

    report: List[dict] = []
    n_blocks = len(blocks)
    for i in range(n_blocks):
        t0 = time.time()
        blk = blocks[i]

        # FP pass on current X (block FP, the rest of the model stays however it was)
        disable_quant(blk)
        Y_fp = _forward_batched(blk, X_cur, device, fwd_chunk)

        # Quant pass on the same X
        enable_quant(blk)
        Y_q  = _forward_batched(blk, X_cur, device, fwd_chunk)

        # LS fit: residual = Y_fp - Y_q ≈ X_cur @ W + b
        Xg = X_cur.reshape(-1, X_cur.size(-1)).to(device)
        Rg = (Y_fp - Y_q).reshape(-1, Y_fp.size(-1)).to(device)
        W, b, r2, abs_err = _linear_regression(Xg, Rg, ridge=ridge)
        rmse_before = (Y_fp - Y_q).pow(2).mean().sqrt().item()
        rmse_after  = (Rg - (Xg @ W + b)).pow(2).mean().sqrt().item()

        linear_init = (i >= start_block)
        new_blk = CompensationBlock(
            W=W.detach().cpu(), b=b.detach().cpu(), r2_score=r2,
            block=blk, linear_init=linear_init, block_id=i,
        ).to(device)
        blocks[i] = new_blk

        # Propagate X_cur through the COMPENSATED block (quant ON, comp ON)
        enable_quant(new_blk)
        X_cur = _forward_batched(new_blk, X_cur, device, fwd_chunk)

        dt = time.time() - t0
        log(f"[qwt] blk {i:2d}  r2={r2:+.4f}  "
            f"rmse {rmse_before:.3e} -> {rmse_after:.3e}  "
            f"init={'linear' if new_blk._init == 'linear' else 'zero'}  "
            f"({dt:.1f}s)")
        report.append({"block": i, "r2": r2, "rmse_before": rmse_before,
                       "rmse_after": rmse_after, "init": new_blk._init, "dt_s": dt})
        del Xg, Rg, W, b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return report


def _forward_batched(block, X, device, chunk):
    outs = []
    for s in range(0, X.size(0), chunk):
        xb = X[s:s + chunk].to(device, non_blocking=True)
        y = block(xb)
        outs.append(y.detach().float().cpu())
    return torch.cat(outs, dim=0)
