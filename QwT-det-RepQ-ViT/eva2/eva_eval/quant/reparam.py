"""RepQ-ViT scale reparameterisation, adapted to EVA-02 ViT block structure.

EVA-02 differences vs EVA-01:
  * Attention uses separate q_proj, k_proj, v_proj (no merged qkv)
  * q_bias / v_bias as separate Parameters (same BEiT-like pattern)
  * FFN is SwiGLU: w1, w2 (parallel branches), then w3
  * SubLN: extra LayerNorm (ffn_ln) between gate and w3

LN -> Linear pairs per EVA-02 Block:
  norm1 -> attn.q_proj, attn.k_proj, attn.v_proj  (one LN feeds three Linears)
  norm2 -> mlp.w1, mlp.w2                          (one LN feeds two Linears)

For norm1, we absorb the per-channel scale into (LN, q_proj) AND (LN, k_proj) AND (LN, v_proj).
Since they share the same LN, we compute a single (r, b) from any one of them (all see the
same activation distribution) and apply it to all three.

For norm2, same idea: w1 and w2 share the same LN output.
"""
import torch
import torch.nn as nn


def collapse_qkv_bias(model):
    """Fold separate q_bias / v_bias into q_proj.bias, v_proj.bias so the
    standard Linear forward produces identical output."""
    n = 0
    for _, m in model.named_modules():
        if hasattr(m, "q_bias") and m.q_bias is not None and hasattr(m, "q_proj"):
            with torch.no_grad():
                if m.q_proj.bias is None:
                    m.q_proj.bias = nn.Parameter(torch.zeros(m.q_proj.out_features,
                                                              device=m.q_bias.device))
                m.q_proj.bias.data.copy_(m.q_bias.data)

                if m.v_proj.bias is None:
                    m.v_proj.bias = nn.Parameter(torch.zeros(m.v_proj.out_features,
                                                              device=m.v_bias.device))
                m.v_proj.bias.data.copy_(m.v_bias.data)

                # k_proj has no bias (by design)
            # After collapsing, the forward should use proj.bias directly
            # Mark as collapsed so forward can skip the F.linear path
            m._bias_collapsed = True
            n += 1
    return n


def _reparam_ln_linear_pair(ln, linear, tag="", verbose=False):
    """Absorb channel-wise activation scale from linear.input_quantizer into (ln, linear).
    Returns True if reparam'd, False if skipped."""
    if not hasattr(linear, "input_quantizer"):
        return False
    iq = linear.input_quantizer
    if not iq.channel_wise or not iq.inited:
        return False

    with torch.no_grad():
        act_delta = iq.delta.reshape(-1)
        act_zp = iq.zero_point.reshape(-1)

        live_mask = act_delta > act_delta.median() * 0.05
        live_delta = act_delta[live_mask]
        live_zp = act_zp[live_mask]

        target_delta = live_delta.mean()
        target_zp = live_zp.mean()
        target_min = -target_zp * target_delta

        r = torch.ones_like(act_delta)
        b = torch.zeros_like(act_delta)
        live_r = live_delta / target_delta
        live_b = -live_zp * live_delta / live_r - target_min
        r[live_mask] = live_r
        b[live_mask] = live_b

        if verbose:
            print(f"  [reparam {tag}] C={act_delta.numel()} "
                  f"live={live_mask.sum().item()}/{act_delta.numel()} "
                  f"r=[{r.min():.3f},{r.max():.3f}] "
                  f"b=[{b.min():.3f},{b.max():.3f}]")

        # Apply to LN
        ln.weight.data = ln.weight.data / r
        ln.bias.data = ln.bias.data / r - b

        # Apply inverse to Linear
        linear.weight.data = linear.weight.data * r
        if linear.bias is None:
            linear.bias = nn.Parameter(torch.zeros(linear.out_features, device=r.device))
        linear.bias.data = linear.bias.data + (linear.weight.data @ b)

        # Switch to per-tensor
        iq.channel_wise = False
        iq.delta = target_delta
        iq.zero_point = target_zp
        linear.weight_quantizer.inited = False

    return True


def scale_reparam_eva2(backbone):
    """For each LN -> {q_proj,k_proj,v_proj} and LN -> {w1,w2}, fold per-channel
    activation scale. Returns count of reparam'd pairs."""
    module_dict = dict(backbone.named_modules())
    n = 0
    verbose_budget = 2

    for name, m in backbone.named_modules():
        cname = type(m).__name__
        if cname not in ("LayerNorm",):
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = module_dict.get(parent_name, None)
        if parent is None:
            continue
        last = name.rsplit(".", 1)[-1]

        verbose = n < verbose_budget

        if last == "norm1":
            # norm1 feeds attn.q_proj, attn.k_proj, attn.v_proj
            attn = getattr(parent, "attn", None)
            if attn is None:
                continue
            # All three see same LN output — reparam LN once using q_proj's scale,
            # then apply same (r, b) to k_proj and v_proj by reparaming them individually
            # but with the ALREADY-modified LN.
            # Simpler approach: reparam each pair. After the first, LN is already modified,
            # so the 2nd and 3rd just apply the inverse to their weight/bias.
            # But this only works if all three have the same channel-wise scale.
            # In practice they do (same input), but to be safe, we reparam only the first
            # (q_proj), and for k_proj/v_proj we just apply the same r to their weights.
            q = getattr(attn, "q_proj", None)
            k = getattr(attn, "k_proj", None)
            v = getattr(attn, "v_proj", None)
            if q is None or not hasattr(q, "input_quantizer"):
                continue
            iq = q.input_quantizer
            if not iq.channel_wise or not iq.inited:
                continue

            with torch.no_grad():
                act_delta = iq.delta.reshape(-1)
                act_zp = iq.zero_point.reshape(-1)
                live_mask = act_delta > act_delta.median() * 0.05
                live_delta = act_delta[live_mask]
                live_zp = act_zp[live_mask]
                target_delta = live_delta.mean()
                target_zp = live_zp.mean()
                target_min = -target_zp * target_delta

                r = torch.ones_like(act_delta)
                b = torch.zeros_like(act_delta)
                live_r = live_delta / target_delta
                live_b = -live_zp * live_delta / live_r - target_min
                r[live_mask] = live_r
                b[live_mask] = live_b

                if verbose:
                    print(f"  [reparam {name}->q/k/v] C={act_delta.numel()} "
                          f"live={live_mask.sum().item()}/{act_delta.numel()} "
                          f"r=[{r.min():.3f},{r.max():.3f}]")

                # Modify LN once
                m.weight.data = m.weight.data / r
                m.bias.data = m.bias.data / r - b

                # Apply inverse to all three projections
                for proj_name, proj in [("q_proj", q), ("k_proj", k), ("v_proj", v)]:
                    if proj is None:
                        continue
                    proj.weight.data = proj.weight.data * r
                    if proj.bias is None:
                        proj.bias = nn.Parameter(torch.zeros(proj.out_features, device=r.device))
                    proj.bias.data = proj.bias.data + (proj.weight.data @ b)

                    if hasattr(proj, "input_quantizer"):
                        proj.input_quantizer.channel_wise = False
                        proj.input_quantizer.delta = target_delta
                        proj.input_quantizer.zero_point = target_zp
                    if hasattr(proj, "weight_quantizer"):
                        proj.weight_quantizer.inited = False
            n += 1

        elif last == "norm2":
            # norm2 feeds mlp.w1 and mlp.w2
            mlp = getattr(parent, "mlp", None)
            if mlp is None:
                continue
            w1 = getattr(mlp, "w1", None)
            w2 = getattr(mlp, "w2", None)
            if w1 is None or not hasattr(w1, "input_quantizer"):
                continue
            iq = w1.input_quantizer
            if not iq.channel_wise or not iq.inited:
                continue

            with torch.no_grad():
                act_delta = iq.delta.reshape(-1)
                act_zp = iq.zero_point.reshape(-1)
                live_mask = act_delta > act_delta.median() * 0.05
                live_delta = act_delta[live_mask]
                live_zp = act_zp[live_mask]
                target_delta = live_delta.mean()
                target_zp = live_zp.mean()
                target_min = -target_zp * target_delta

                r = torch.ones_like(act_delta)
                b = torch.zeros_like(act_delta)
                live_r = live_delta / target_delta
                live_b = -live_zp * live_delta / live_r - target_min
                r[live_mask] = live_r
                b[live_mask] = live_b

                if verbose:
                    print(f"  [reparam {name}->w1/w2] C={act_delta.numel()} "
                          f"live={live_mask.sum().item()}/{act_delta.numel()} "
                          f"r=[{r.min():.3f},{r.max():.3f}]")

                # Modify LN once
                m.weight.data = m.weight.data / r
                m.bias.data = m.bias.data / r - b

                # Apply inverse to both w1 and w2
                for proj in [w1, w2]:
                    if proj is None:
                        continue
                    proj.weight.data = proj.weight.data * r
                    if proj.bias is None:
                        proj.bias = nn.Parameter(torch.zeros(proj.out_features, device=r.device))
                    proj.bias.data = proj.bias.data + (proj.weight.data @ b)

                    if hasattr(proj, "input_quantizer"):
                        proj.input_quantizer.channel_wise = False
                        proj.input_quantizer.delta = target_delta
                        proj.input_quantizer.zero_point = target_zp
                    if hasattr(proj, "weight_quantizer"):
                        proj.weight_quantizer.inited = False
            n += 1

        elif last == "ffn_ln":
            # ffn_ln (SubLN inside SwiGLU) feeds w3
            w3 = getattr(parent, "w3", None)
            if w3 is None or not hasattr(w3, "input_quantizer"):
                continue
            iq = w3.input_quantizer
            if not iq.channel_wise or not iq.inited:
                continue

            if _reparam_ln_linear_pair(m, w3, tag=f"{name}->w3", verbose=verbose):
                n += 1

    return n