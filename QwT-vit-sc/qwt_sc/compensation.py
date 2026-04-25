"""QwT-style linear residual compensation for stochastic-computing ViTs.

Given two copies of a transformer:
  * ``model_fp`` — full-precision reference
  * ``model_sc`` — with SC matmuls patched into its attention / MLP path

we fit, per block ``i``, a small closed-form residual

    (Y_fp_i - Y_sc_i)  ~=  X_i @ W_i + b_i

on a small calibration set, then wrap each SC block as

    out_i(x) = block_sc_i(x) + x @ W_i + b_i.

This is the direct SC-analogue of the ``CompensationBlock`` in
``QwT-cls-RepQ-ViT/qwt_vit_and_deit.py`` (Fu et al., CVPR 2025): no back-prop,
closed-form LS, calibration in minutes. The routine is kernel-agnostic — any
SC backend (XNOR/Sobol bitstream, calibrated Gaussian surrogate, etc.) is
observed only through ``model_sc``'s forward.

--------------------------------------------------------------------------
The admission algorithm: cross-seed cosine gate
--------------------------------------------------------------------------

The residual ``R_i = Y_fp_i - Y_sc_i`` contains two additive components:

    R_i(X) = bias_i(X)   +   noise_i(X)
             \_____/          \______/
             deterministic     input-dependent SC noise:
             part of the       depends on the specific
             SC->FP gap        (X, Sobol-state) combination

Ridge LS on a single calib batch fits a single ``W_i`` that captures both
components indistinguishably. When ``bias_i`` is *small relative to*
``noise_i`` (the common case at high-baseline configs such as avg192, where
the raw SC top-1 is already 84.7%), the LS solution is dominated by the
noise component. At eval time the image distribution is different (50k vs
1024 images, different token-level activations), the noise pattern differs,
and ``W_i`` amplifies in a now-orthogonal direction. This "noise-fit" ``W``
does not compensate — it injects correlated error that compounds through
every downstream block; a single noise-fit block near the input can destroy
the pre-head embedding and drop top-1 to 0%.

The cross-seed gate discriminates signal from noise by fitting **two**
independent ``W``'s on two disjoint calibration batches and admitting the
block only when they agree in direction:

    cos(W_A, W_B) := <flatten(W_A), flatten(W_B)> / (||W_A|| * ||W_B||)

Physical meaning: a bias direction that is shared across two different
image populations survives the change — cos ≈ 1. A noise-fit direction is
specific to the particular images drawn, so two different draws produce
orthogonal ``W`` — cos ≈ 0. The gap is large and regime-invariant.

Admission rule, applied to every block:

    admit iff  cos(W_A, W_B) > τ  AND  min(||W_A||, ||W_B||) > ε

Install  ``W̄ = (W_A + W_B)/2``  and  ``b̄ = (b_A + b_B)/2``.

--------------------------------------------------------------------------
Why it supersedes the legacy heuristic gates
--------------------------------------------------------------------------

Before cross-seed, admission was gated by a combination of r²-threshold,
cv-holdout r², rmse_before_floor, max_late_blocks (zonal cap), rmse_margin,
last_block_r2_threshold, and rmse_before_override. Each of these is a proxy
for "is this W signal or noise?", and each proxy leaked on one regime:

  * r² > 0.4 on held-out rows: block 1 at avg192 scores 0.54 and is
    admitted, yet generalizes terribly (the held-out rows share the
    calib batch's noise pattern). Collapse.
  * r² > 0.5 (opt3): by luck excludes block 1 on avg192 (in-sample r² = 0.40),
    works on that regime but was manually discovered.
  * max_late_blocks = 2 (V5): protects the mid-zone at p7 but does not
    prevent early-block collapse (block 1, block 6) at avg192.
  * rmse_before_floor: blunt — a floor that saves avg192 also rejects
    p7's high-signal early blocks.

Cross-seed is the physically correct test. It admits exactly those blocks
where the fit survives an input-population shift, which is the same shift
that happens between calib (1024 images) and eval (50k).

--------------------------------------------------------------------------
N=50k production results (5 configs, ``--cos_threshold 0.5``,
``--last_block_cos_threshold 0.8``, ``--lookahead_veto``, **SC comp**)
--------------------------------------------------------------------------

Sweep ``cls/results/sweep_2026-04-25/`` (cross-seed gate + B_ha
``HeadAlignedSCLinear`` comp kernel, ``n_heads=16``, ``sc_prec=8``,
``bipolar``; fresh MP search at ``SEARCH_N_SEARCH=64``,
``SEARCH_MAX_ITERS=12``):

    config           raw SC   B_ha+gate   Δ      legacy r²-gate Δ
    p7_uniform        78.94      82.87  +3.93         +3.12
    p7_mp             79.89      83.26  +3.37         +1.14
    p8_uniform        85.57      85.70  +0.13         +0.20
    avg192_uniform    84.67      85.42  +0.75         +0.67
    avg192_mp         84.62      85.47  +0.85         +0.65
    geomean Δ                          +1.80         +1.16

**Cross-seed gate admits 22/24 blocks** on every regime — blocks 0 and
23 rejected, blocks 1-22 admitted. The bimodal structure of per-block
``cos_ab`` is clean: rejected blocks sit near 0 (||W|| ≫ 0 but the
cross-seed cosine collapses), admitted blocks span [+0.7, +0.95], and
the gap between the two modes is ~0.7. ``τ`` is insensitive inside
[0.3, 0.65].

The B_ha kernel beats raw SC on every config and beats the legacy
r²-gate + SC-comp reference on every config. The +0.64 geomean lift
over the legacy reference comes jointly from (a) the cross-seed gate
itself and (b) the wider-budget fresh MP search; the dominant config
contributing is ``p7_mp`` (+1.14 → +3.37, a +2.23 pt jump from the
larger search budget on the same proj/mlp search space).

See ``docs/SC_COMP_ALGORITHM.md`` for the full kernel-choice writeup
including the A_fw vs B_ha N=1000 pilot, hardware-cost accounting, and
the next-iteration recommendations on the MP search.

--------------------------------------------------------------------------
Tuning τ
--------------------------------------------------------------------------

Recommended default: ``cos_threshold = 0.5``, ``last_block_cos_threshold =
0.8`` (block 23 feeds the pre-head embedding directly, so demand tighter
agreement there), ``norm_floor = 0.0``. Tighten τ toward 0.65 if a regime
admits borderline blocks that drag top-1; loosen toward 0.3 if the gate
rejects clearly-signal blocks (unlikely given the current gap).

--------------------------------------------------------------------------
Cost
--------------------------------------------------------------------------

Two SC forwards and two ridge solves per block during calibration; in
practice ~1.5-2× the legacy single-batch calib wall time. Negligible
compared to N=50k evaluation. Eval path is unchanged (single installed W̄
per admitted block).
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn


class CompensationBlock(nn.Module):
    """Residual linear correction wrapping an inner block.

    ``out = block(x) + x @ W + b``, with ``W`` of shape ``(D_in, D_out)``.
    The correction is an ``nn.Linear`` so trailing-dim inputs ``(B, N, D_in)``
    are handled naturally across token dimensions.

    Args:
        block: the inner block (quantized / SC / anything).
        W: ``(D_in, D_out)`` tensor from the LS fit.
        b: ``(D_out,)`` tensor.
        cos_ab: cross-seed cosine from the admission gate (diagnostic).
        enabled: if ``False``, the wrapper is a no-op — useful when the
            admission rule rejected the block.
    """

    def __init__(self, block: nn.Module, W: torch.Tensor, b: torch.Tensor,
                 cos_ab: float = 0.0, enabled: bool = True,
                 comp_module: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        if comp_module is not None:
            self.comp = comp_module
        else:
            D_in, D_out = W.shape
            self.comp = nn.Linear(D_in, D_out, bias=True)
            with torch.no_grad():
                self.comp.weight.copy_(W.t().contiguous())
                self.comp.bias.copy_(b)
        self.cos_ab = float(cos_ab)
        self.enabled = bool(enabled)

    def forward(self, x, *args, **kwargs):
        out = self.block(x, *args, **kwargs)
        if not self.enabled:
            return out
        return out + self.comp(x)


def closed_form_ridge(X: torch.Tensor, Y: torch.Tensor,
                      ridge: float = 1e-2,
                      ) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Solve ``min_{W,b} ||Y - (X W + b)||^2 + ridge * ||W||^2`` in closed form.

    Stacks an all-ones column to fit the bias; ridges only the slope.
    Returns ``(W, b, r2)`` where r² is the in-sample coefficient of
    determination — diagnostic only under the cross-seed gate (the actual
    admission test is ``cos(W_A, W_B)``, not r²).
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
    calib_loader_a,
    calib_loader_b,
    device: torch.device,
    n_calib: int,
    ridge: float = 1e-2,
    start_block: int = 0,
    fwd_chunk: int = 32,
    cos_threshold: float = 0.5,
    norm_floor: float = 0.0,
    last_block_cos_threshold: Optional[float] = None,
    lookahead_veto: bool = False,
    comp_factory: Optional[Callable[[torch.Tensor, torch.Tensor], nn.Module]] = None,
    comp_factory_variants: Optional[List[tuple]] = None,
    log_fn: Callable[[str], None] = print,
) -> List[dict]:
    """Cross-seed QwT calibration — installs a per-block comp iff ``W`` is
    reproducible across two disjoint calib batches.

    For each block ``i``:

      1. Fit ``W_A, b_A`` from batch A's ``(X_A, R_A)`` where
         ``R_A = Y_fp(X_A) - Y_sc(X_A)``.
      2. Fit ``W_B, b_B`` from batch B's ``(X_B, R_B)``.
      3. Admit iff ``cos(flatten(W_A), flatten(W_B)) > cos_threshold`` AND
         ``min(||W_A||, ||W_B||) > norm_floor``.
      4. Install ``W̄ = (W_A + W_B)/2``, ``b̄ = (b_A + b_B)/2``.
      5. Propagate both chains ``X_A, X_B`` through the installed wrapper so
         block ``i+1`` sees the post-comp distribution (Gauss-Seidel).

    Args:
        model_fp: FP reference; used only to feed images for batch collection.
        model_sc: SC-patched model; mutated in place (blocks wrapped).
        blocks_fp, blocks_sc_container: parallel block lists.
        calib_loader_a, calib_loader_b: two DataLoaders with disjoint image
            samples (different seeds). Must each yield at least ``n_calib``
            images. Their disjointness is what makes the cosine test work.
        n_calib: samples to use per batch.
        ridge: ℓ2 regularization on W in each LS fit.
        start_block: indices < start_block are never admitted (pass-through).
        fwd_chunk: forward batch size for block-level forwards.
        cos_threshold: admission threshold τ. See module docstring for
            tuning guidance; 0.5 is the recommended default.
        norm_floor: additionally require min(||W_A||, ||W_B||) > this. 0.0
            disables; useful when some blocks have ≈0 bias and cosine is
            numerically noisy.
        last_block_cos_threshold: stricter τ for the last block only.
            Block 23 feeds the pre-head embedding directly, so demand
            tighter agreement. 0.8 is the recommended default; ``None``
            uses ``cos_threshold``.
        lookahead_veto: one-step binary lookahead — veto apply if propagating
            through block ``i+1`` shows 'skip' is closer to the FP chain than
            'apply'. Uses batch A for the check. Conservative; rarely fires
            under the cosine gate but is cheap insurance.
        log_fn: logging callback.

    Returns:
        List of per-block dicts with keys ``block, enabled, cos_ab,
        cos_threshold_used, norm_a, norm_b, norm_floor, r2_a, r2_b,
        rmse_before_a, rmse_before_b, rmse_after_a, rmse_after_b,
        lookahead_decision, lookahead_err_apply, lookahead_err_skip, dt_s``.
    """
    assert len(blocks_fp) == len(blocks_sc_container), (
        f"fp blocks ({len(blocks_fp)}) vs sc blocks "
        f"({len(blocks_sc_container)}) length mismatch"
    )
    n_blocks = len(blocks_fp)

    def _collect_x0(loader, tag):
        captured: List[torch.Tensor] = []
        first_block = blocks_sc_container[0]

        def hook(_m, args):
            captured.append(args[0].detach().float().cpu())

        log_fn(f"[calib] collecting block-0 inputs for batch {tag}")
        handle = first_block.register_forward_pre_hook(hook)
        seen = 0
        try:
            for batch in loader:
                imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
                imgs = imgs.to(device, non_blocking=True)
                model_sc(imgs)
                seen += imgs.size(0)
                if seen >= n_calib:
                    break
        finally:
            handle.remove()
        return torch.cat(captured, dim=0)[:n_calib]

    X_a = _collect_x0(calib_loader_a, "A")
    X_b = _collect_x0(calib_loader_b, "B")
    log_fn(f"[calib] X_a {tuple(X_a.shape)}  X_b {tuple(X_b.shape)}")

    report: List[dict] = []
    for i in range(n_blocks):
        t0 = time.time()
        blk_fp = blocks_fp[i]
        blk_sc = blocks_sc_container[i]

        Y_fp_a = _forward_batched(blk_fp, X_a, device, fwd_chunk)
        Y_sc_a = _forward_batched(blk_sc, X_a, device, fwd_chunk)
        Y_fp_b = _forward_batched(blk_fp, X_b, device, fwd_chunk)
        Y_sc_b = _forward_batched(blk_sc, X_b, device, fwd_chunk)

        Xa_flat = X_a.reshape(-1, X_a.size(-1)).to(device)
        Ra_flat = (Y_fp_a - Y_sc_a).reshape(-1, Y_fp_a.size(-1)).to(device)
        Xb_flat = X_b.reshape(-1, X_b.size(-1)).to(device)
        Rb_flat = (Y_fp_b - Y_sc_b).reshape(-1, Y_fp_b.size(-1)).to(device)

        W_a, b_a, r2_a = closed_form_ridge(Xa_flat, Ra_flat, ridge=ridge)
        W_b, b_b, r2_b = closed_form_ridge(Xb_flat, Rb_flat, ridge=ridge)

        rmse_before_a = (Y_fp_a - Y_sc_a).pow(2).mean().sqrt().item()
        rmse_before_b = (Y_fp_b - Y_sc_b).pow(2).mean().sqrt().item()

        # Cosine of flattened W matrices. Bias is excluded — a shared offset
        # is always "reproducible" and would dilute the signal. Biases get
        # averaged for install.
        wa_vec = W_a.flatten()
        wb_vec = W_b.flatten()
        na = wa_vec.norm().item()
        nb = wb_vec.norm().item()
        cos_ab = (float((wa_vec @ wb_vec).item()) / (na * nb)) if (na > 0 and nb > 0) else 0.0

        W = (W_a + W_b) / 2.0
        b = (b_a + b_b) / 2.0

        rmse_after_a = (Ra_flat - (Xa_flat @ W + b)).pow(2).mean().sqrt().item()
        rmse_after_b = (Rb_flat - (Xb_flat @ W + b)).pow(2).mean().sqrt().item()

        thr_i = cos_threshold
        if last_block_cos_threshold is not None and i == n_blocks - 1:
            thr_i = max(thr_i, last_block_cos_threshold)

        norm_ok = min(na, nb) > norm_floor
        cos_ok = cos_ab > thr_i
        enabled = (i >= start_block) and cos_ok and norm_ok

        # One-step lookahead on batch A: does propagating through blk_fp[i+1]
        # under 'apply' beat 'skip'? Veto apply if not. Cheap, rarely fires
        # under the cosine gate but provides a principled tiebreaker for the
        # borderline-cos regime.
        lookahead_err_apply = None
        lookahead_err_skip = None
        lookahead_decision = None
        if enabled and lookahead_veto and (i + 1 < n_blocks):
            delta_gpu = (Xa_flat @ W + b).reshape(Y_sc_a.shape).float()
            Y_apply = Y_sc_a + delta_gpu.cpu()
            blk_fp_next = blocks_fp[i + 1]
            Y_ref_next = _forward_batched(blk_fp_next, Y_fp_a, device, fwd_chunk)
            Y_apply_next = _forward_batched(blk_fp_next, Y_apply, device, fwd_chunk)
            Y_skip_next = _forward_batched(blk_fp_next, Y_sc_a, device, fwd_chunk)
            lookahead_err_apply = (Y_ref_next - Y_apply_next).pow(2).mean().item()
            lookahead_err_skip = (Y_ref_next - Y_skip_next).pow(2).mean().item()
            del delta_gpu, Y_apply, Y_ref_next, Y_apply_next, Y_skip_next
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if lookahead_err_skip < lookahead_err_apply:
                lookahead_decision = "veto"
                enabled = False
                log_fn(f"[lookahead] block {i}: VETO apply "
                       f"(err_apply={lookahead_err_apply:.4e} > err_skip={lookahead_err_skip:.4e})")
            else:
                lookahead_decision = "keep"

        W_cpu = W.detach().cpu()
        b_cpu = b.detach().cpu()

        # Build the comp kernel. The cross-seed gate above decides *whether*
        # to apply a correction; `comp_factory` / `comp_factory_variants`
        # decide *how* the correction matmul runs (SCLinear, head-aligned,
        # FP nn.Linear, …). When `comp_factory_variants` is given, each
        # candidate kernel is measured against the calib residual and the
        # one with lowest RMSE wins (per-block pick = O(log K) config bits
        # in hardware). When only `comp_factory` is given it's a fast-path
        # single kernel. When both are None, the CompensationBlock falls
        # back to an FP nn.Linear — the historical debug path.
        chosen_variant = None
        variant_rmses = {}
        comp_module = None

        def _measure_variant(mod):
            mod = mod.to(device)
            with torch.no_grad():
                cs = []
                chunk_rows = max(fwd_chunk * 257, 1)
                for s in range(0, Xa_flat.size(0), chunk_rows):
                    e = min(s + chunk_rows, Xa_flat.size(0))
                    cs.append(mod(Xa_flat[s:e]))
                c_act = torch.cat(cs, dim=0)
            return (Ra_flat - c_act).pow(2).mean().sqrt().item()

        if comp_factory_variants is not None and len(comp_factory_variants) > 0:
            best = None
            for vname, vfac in comp_factory_variants:
                cand = vfac(W_cpu, b_cpu)
                rmse_v = _measure_variant(cand)
                variant_rmses[vname] = rmse_v
                if best is None or rmse_v < best[0]:
                    best = (rmse_v, vname, cand)
            _, chosen_variant, comp_module = best
        elif comp_factory is not None:
            comp_module = comp_factory(W_cpu, b_cpu)

        new_block = CompensationBlock(
            block=blk_sc, W=W_cpu, b=b_cpu,
            cos_ab=cos_ab, enabled=enabled, comp_module=comp_module,
        ).to(device)
        blocks_sc_container[i] = new_block

        # Propagate both chains through the installed wrapper (Gauss-Seidel).
        X_a = _forward_batched(new_block, X_a, device, fwd_chunk)
        X_b = _forward_batched(new_block, X_b, device, fwd_chunk)

        dt = time.time() - t0
        var_str = f"  comp={chosen_variant}" if chosen_variant is not None else ""
        log_fn(
            f"[calib] block {i:2d}  cos={cos_ab:+.4f}  "
            f"||W_a||={na:.2e} ||W_b||={nb:.2e}  "
            f"r2 {r2_a:+.3f}/{r2_b:+.3f}  "
            f"rmse_before {rmse_before_a:.4e}/{rmse_before_b:.4e}  "
            f"enabled={enabled}{var_str}  ({dt:.1f}s)",
        )
        report.append({
            "block": i, "enabled": enabled,
            "cos_ab": cos_ab,
            "cos_threshold_used": thr_i,
            "norm_a": na, "norm_b": nb, "norm_floor": norm_floor,
            "r2_a": r2_a, "r2_b": r2_b,
            "rmse_before_a": rmse_before_a, "rmse_before_b": rmse_before_b,
            "rmse_after_a": rmse_after_a, "rmse_after_b": rmse_after_b,
            "lookahead_decision": lookahead_decision,
            "lookahead_err_apply": lookahead_err_apply,
            "lookahead_err_skip": lookahead_err_skip,
            "variant": chosen_variant,
            "variant_rmses": variant_rmses,
            "dt_s": dt,
        })

        del Xa_flat, Xb_flat, Ra_flat, Rb_flat, W_a, b_a, W_b, b_b, W, b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return report
