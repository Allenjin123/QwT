"""RepQ-ViT scale reparameterisation, adapted to EVA-01 ViT block structure.

Idea: LayerNorm output has severe channel-wise variance (outlier channels). Instead of
quantising activations channel-wise (awkward for matmul), we absorb the per-channel
scale into the LayerNorm's (weight, bias) and the inverse into the next Linear's
(weight, bias). After reparam the activation distribution is flat across channels,
so per-tensor quantisation works.

Two LN -> Linear pairs per EVA Block:
  norm1 -> attn.qkv
  norm2 -> mlp.fc1

EVA wrinkle: beit_like_qkv_bias=True synthesises the qkv bias as concat(q_bias, 0, v_bias)
in forward, ignoring self.qkv.bias. We collapse q_bias/v_bias into self.qkv.bias first
and disable beit_like so the reparam bias compensation (W @ b) can land in the standard
bias slot.
"""
import torch
import torch.nn as nn


def collapse_beit_like_qkv_bias(model):
    """Fold separate q_bias / v_bias into self.qkv.bias so the standard Linear forward
    produces identical output, then turn beit_like off."""
    n = 0
    for _, m in model.named_modules():
        if getattr(m, "beit_like_qkv_bias", False):
            with torch.no_grad():
                eff = torch.cat([m.q_bias, torch.zeros_like(m.v_bias), m.v_bias])
                if m.qkv.bias is None:
                    m.qkv.bias = nn.Parameter(torch.zeros(m.qkv.out_features, device=eff.device))
                m.qkv.bias.data.copy_(eff)
            m.beit_like_qkv_bias = False
            n += 1
    return n


def scale_reparam_eva(backbone):
    """For each LN whose sibling Linear has an *inited, channel-wise* input quantiser,
    fold the per-channel activation scale into (LN, Linear) and switch the Linear's
    input quantiser to per-tensor. Returns count of reparam'd pairs."""
    module_dict = dict(backbone.named_modules())
    n = 0
    for name, m in backbone.named_modules():
        cname = type(m).__name__
        if cname not in ("LayerNorm", "LayerNormWithForceFP32"):
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = module_dict.get(parent_name, None)
        if parent is None:
            continue
        last = name.rsplit(".", 1)[-1]
        if last == "norm1":
            nxt = getattr(getattr(parent, "attn", None), "qkv", None)
        elif last == "norm2":
            nxt = getattr(getattr(parent, "mlp", None), "fc1", None)
        else:
            continue
        if nxt is None or not hasattr(nxt, "input_quantizer"):
            continue
        iq = nxt.input_quantizer
        if not iq.channel_wise or not iq.inited:
            continue

        with torch.no_grad():
            act_delta = iq.delta.reshape(-1)
            act_zp    = iq.zero_point.reshape(-1)
            act_min   = -act_zp * act_delta

            # Dead channel mask: channels whose activation range is << median.
            # Their per-channel delta ~ 0 would cause LN.weight /= 0 blow-ups,
            # AND they'd bias target_delta downward. Skip them: leave r=1, b=0
            # so their LN/Linear slice is untouched. The per-tensor quantiser
            # will absorb the small residual from those channels fine.
            live_mask = act_delta > act_delta.median() * 0.05    # (C,) bool
            live_delta = act_delta[live_mask]
            live_zp    = act_zp[live_mask]

            target_delta = live_delta.mean()
            target_zp    = live_zp.mean()
            target_min   = -target_zp * target_delta

            # Default r=1, b=0 (no change); fill only live channels
            r = torch.ones_like(act_delta)
            b = torch.zeros_like(act_delta)
            live_r = live_delta / target_delta
            live_b = -live_zp * live_delta / live_r - target_min   # = act_min/r - target_min
            r[live_mask] = live_r
            b[live_mask] = live_b
            if n < 2:
                print(f"  [reparam {name}] C={act_delta.numel()} "
                      f"live={live_mask.sum().item()}/{act_delta.numel()} "
                      f"r=[{r.min():.3f},{r.max():.3f}] "
                      f"b=[{b.min():.3f},{b.max():.3f}]")

            m.weight.data = m.weight.data / r
            m.bias.data   = m.bias.data / r - b

            nxt.weight.data = nxt.weight.data * r
            if nxt.bias is None:
                nxt.bias = nn.Parameter(torch.zeros(nxt.out_features, device=r.device))
            nxt.bias.data = nxt.bias.data + (nxt.weight.data @ b)

            iq.channel_wise = False
            iq.delta = target_delta
            iq.zero_point = target_zp
            nxt.weight_quantizer.inited = False      # re-init on next forward
        n += 1
    return n
