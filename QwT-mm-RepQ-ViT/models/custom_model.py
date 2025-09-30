import torch
import torch.nn as nn
from typing import Final, Optional, Callable
from . import model as original_model
from termcolor import colored
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

def attention_forward(self, x: torch.Tensor) -> torch.Tensor:
    if not self.batch_first:
        x = x.transpose(0, 1)

    B, N, C = x.shape  

    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    q = q * self.scale
    # attn = q @ k.transpose(-2, -1)
    attn = self.matmul1(q, k.transpose(-2, -1))
    attn = attn.softmax(dim=-1)
    x = self.matmul2(attn, v)

    x = x.transpose(1, 2).reshape(B, N, C)

    x = self.proj(x)

    if not self.batch_first:
        x = x.transpose(0, 1)

    return x

class ModelWithClassifier(nn.Module):
    def __init__(self, model, output_dim):
        super(ModelWithClassifier, self).__init__()
        self.model = model 
        self.classifier = nn.Linear(model.visual.output_dim, output_dim)

    def forward(self, x):
        features = self.model.encode_image(x)
        features = features / features.norm(dim=1, keepdim=True)
        logits = self.classifier(features)
        return logits
    
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
            batch_first: bool = False
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False
        self.batch_first = batch_first

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.batch_first:
            x = x.transpose(0, 1)

        B, N, C = x.shape 

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) 

        q = q * self.scale
        # attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn_mask + attn
        
        attn = attn.softmax(dim=-1)
        x = attn @ v


        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)

        if not self.batch_first:
            x = x.transpose(0, 1)

        return x

import torch
import torch.nn as nn
from collections import OrderedDict

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, activation: nn.Module = QuickGELU(),
            batch_first: bool = False
    ):
        super().__init__()

        self.ln_1 = LayerNorm(d_model)
        
        self.attn = Attention(
            dim=d_model,
            num_heads=n_head,
            batch_first=batch_first
        )

        self.ln_2 = LayerNorm(d_model)
        
        mlp_ratio = 4        
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", activation),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, self.attn_mask)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        return x


def replace_resblocks_with_vision_transformer(model: nn.Module):
    num = 0
    replacements = []

    for name, module in model.visual.transformer.resblocks.named_children():
        if isinstance(module, original_model.ResidualAttentionBlock):
            new_module = ResidualAttentionBlock(
                d_model=module.attn.embed_dim, 
                n_head=module.attn.num_heads
            )

            new_module.attn.qkv.weight.data.copy_(module.attn.in_proj_weight.data)
            new_module.attn.qkv.bias.data.copy_(module.attn.in_proj_bias.data)
            new_module.attn.proj.weight.data.copy_(module.attn.out_proj.weight.data)
            new_module.attn.proj.bias.data.copy_(module.attn.out_proj.bias.data)

            new_module.ln_1.weight.data.copy_(module.ln_1.weight.data)
            new_module.ln_1.bias.data.copy_(module.ln_1.bias.data)
            new_module.ln_2.weight.data.copy_(module.ln_2.weight.data)
            new_module.ln_2.bias.data.copy_(module.ln_2.bias.data)

            new_module.mlp[0].weight.data.copy_(module.mlp[0].weight.data)  # c_fc
            new_module.mlp[0].bias.data.copy_(module.mlp[0].bias.data)      # c_fc
            new_module.mlp[2].weight.data.copy_(module.mlp[2].weight.data)  # c_proj
            new_module.mlp[2].bias.data.copy_(module.mlp[2].bias.data)      # c_proj


            replacements.append((name, new_module))
            num += 1
    assert num > 0, "No ResidualAttentionBlock found in model.visual.transformer.resblocks"

    for name, new_module in replacements:
        setattr(model.visual.transformer.resblocks, name, new_module)

    print(colored(f"Replaced {num} ResidualAttentionBlock modules in model.visual.transformer.resblocks.", "green"))
    return model
    
    
def replace_resblocks_with_transformer(model: nn.Module):
    num = 0
    replacements = []

    for name, module in model.transformer.resblocks.named_children():
        if isinstance(module, original_model.ResidualAttentionBlock):
            new_module = ResidualAttentionBlock(
                d_model=module.attn.embed_dim, 
                n_head=module.attn.num_heads,
                attn_mask=model.build_attention_mask()
            )

            new_module.attn.qkv.weight.data.copy_(module.attn.in_proj_weight.data)
            new_module.attn.qkv.bias.data.copy_(module.attn.in_proj_bias.data)
            new_module.attn.proj.weight.data.copy_(module.attn.out_proj.weight.data)
            new_module.attn.proj.bias.data.copy_(module.attn.out_proj.bias.data)

            new_module.ln_1.weight.data.copy_(module.ln_1.weight.data)
            new_module.ln_1.bias.data.copy_(module.ln_1.bias.data)
            new_module.ln_2.weight.data.copy_(module.ln_2.weight.data)
            new_module.ln_2.bias.data.copy_(module.ln_2.bias.data)

            new_module.mlp[0].weight.data.copy_(module.mlp[0].weight.data)  # c_fc
            new_module.mlp[0].bias.data.copy_(module.mlp[0].bias.data)      # c_fc
            new_module.mlp[2].weight.data.copy_(module.mlp[2].weight.data)  # c_proj
            new_module.mlp[2].bias.data.copy_(module.mlp[2].bias.data)      # c_proj
            new_module.attn_mask = module.attn_mask


            replacements.append((name, new_module))
            num += 1
    assert num > 0, "No ResidualAttentionBlock found in model.transformer.resblocks"

    for name, new_module in replacements:
        setattr(model.transformer.resblocks, name, new_module)
    print(colored(f"Replaced {num} ResidualAttentionBlock modules in model.transformer.resblocks.", "green"))
    return model