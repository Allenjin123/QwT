"""Walk an EVA ViTDet model and swap nn.Conv2d / nn.Linear / MatMul with quantised
wrappers. Naming rules inherited from RepQ-ViT:
  - 'embed' in name  -> input quantiser gets n_bits=8 (patch embedding is sensitive)
  - 'qkv' / 'fc1'    -> channel-wise input quantisation
  - 'matmul2'        -> log-sqrt(2) quantisation for post-softmax matmul
"""
from copy import deepcopy
import torch.nn as nn

from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul, MatMul


def quant_model_eva(model, input_quant_params=None, weight_quant_params=None):
    input_quant_params = input_quant_params or {}
    weight_quant_params = weight_quant_params or {}

    # patch-embed conv: keep 8-bit activations
    input_quant_params_embed = deepcopy(input_quant_params)
    input_quant_params_embed["n_bits"] = 8

    # post-softmax matmul (attn @ v): log-sqrt(2) quantiser
    input_quant_params_matmul2 = deepcopy(input_quant_params)
    input_quant_params_matmul2["log_quant"] = True

    # SimQuant: channel-wise for qkv / fc1 inputs
    input_quant_params_channel = deepcopy(input_quant_params)
    input_quant_params_channel["channel_wise"] = True

    # build name -> module map for parent lookup
    module_dict = {name: m for name, m in model.named_modules()}

    for name, m in list(model.named_modules()):
        parent_name, _, attr = name.rpartition(".")
        parent = module_dict.get(parent_name, model if parent_name == "" else None)
        if parent is None:
            continue

        # exact-type only: skip detectron2.layers.Conv2d wrapper (has norm/activation)
        if type(m) is nn.Conv2d:
            in_q = input_quant_params_embed if "embed" in name else input_quant_params
            new_m = QuantConv2d(
                m.in_channels, m.out_channels, m.kernel_size,
                m.stride, m.padding, m.dilation, m.groups,
                bias=m.bias is not None,
                input_quant_params=in_q,
                weight_quant_params=weight_quant_params,
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(parent, attr, new_m)

        elif type(m) is nn.Linear:
            sim_quant = any(k in name for k in ("qkv", "fc1"))
            in_q = input_quant_params_channel if sim_quant else input_quant_params
            new_m = QuantLinear(
                m.in_features, m.out_features,
                input_quant_params=in_q,
                weight_quant_params=weight_quant_params,
            )
            new_m.weight.data = m.weight.data
            new_m.bias = m.bias
            setattr(parent, attr, new_m)

        elif m.__class__.__name__ == "MatMul" and not isinstance(m, QuantMatMul):
            # Match by class name so any MatMul marker (ours or EVA's vit.MatMul) is caught
            in_q = input_quant_params_matmul2 if "matmul2" in name else input_quant_params
            new_m = QuantMatMul(input_quant_params=in_q)
            setattr(parent, attr, new_m)

    return model


def set_quant_state(model, input_quant=False, weight_quant=False):
    for m in model.modules():
        if isinstance(m, (QuantConv2d, QuantLinear, QuantMatMul)):
            m.set_quant_state(input_quant, weight_quant)
