import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(UniformQuantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
    
    def __repr__(self):
        s = super(UniformQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # start quantization
        x_int = torch.round(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            # Vectorised per-channel percentile search (mathematically equivalent
            # to the original Python for-loop, but ~10-100x faster on EVA-L).
            x_clone = x.clone().detach()
            if x.dim() == 3:
                x_flat = x_clone.permute(2, 0, 1).reshape(x.shape[-1], -1)
                out_shape = (1, 1, -1)
            elif x.dim() == 2:
                x_flat = x_clone.reshape(x.shape[0], -1)
                out_shape = (-1, 1)
            elif x.dim() == 4:
                x_flat = x_clone.reshape(x.shape[0], -1)
                out_shape = (-1, 1, 1, 1)
            else:
                raise NotImplementedError

            C = x_flat.shape[0]
            n_levels = 2 ** self.n_bits
            best_score = torch.full((C,), float('inf'),
                                    device=x_flat.device, dtype=x_flat.dtype)
            delta = torch.zeros(C, device=x_flat.device, dtype=x_flat.dtype)
            zero_point = torch.zeros(C, device=x_flat.device, dtype=x_flat.dtype)

            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    q_hi = torch.quantile(x_flat, pct, dim=1)
                    q_lo = torch.quantile(x_flat, 1.0 - pct, dim=1)
                except RuntimeError:
                    # torch.quantile has a ~16M element per-call limit; numpy fallback.
                    x_cpu = x_flat.detach().cpu().numpy()
                    q_hi = torch.tensor(np.percentile(x_cpu, pct * 100, axis=1),
                                        device=x_flat.device, dtype=x_flat.dtype)
                    q_lo = torch.tensor(np.percentile(x_cpu, (1 - pct) * 100, axis=1),
                                        device=x_flat.device, dtype=x_flat.dtype)

                delta_try = (q_hi - q_lo) / (n_levels - 1)
                # guard degenerate (constant) channels so fake-quant doesn't div-by-zero
                delta_safe = torch.where(delta_try > 0, delta_try, torch.ones_like(delta_try))
                zp_try = (-q_lo / delta_safe).round()

                x_int = torch.round(x_flat / delta_safe.unsqueeze(1))
                x_q = torch.clamp(x_int + zp_try.unsqueeze(1), 0, n_levels - 1)
                x_dq = (x_q - zp_try.unsqueeze(1)) * delta_safe.unsqueeze(1)

                score = (x_flat - x_dq).abs().pow(2).mean(dim=1)
                better = score < best_score
                best_score = torch.where(better, score, best_score)
                delta = torch.where(better, delta_try, delta)
                zero_point = torch.where(better, zp_try, zero_point)

            delta = delta.view(*out_shape)
            zero_point = zero_point.view(*out_shape)
        else:
            x_clone = x.clone().detach()
            x_max = x_clone.max()
            x_min = x_clone.min()
            best_score = 1e+10
            for pct in [0.999, 0.9999, 0.99999]:
                try:
                    new_max = torch.quantile(x_clone.reshape(-1), pct)
                    new_min = torch.quantile(x_clone.reshape(-1), 1.0 - pct)
                except:
                    new_max = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), pct * 100),
                        device=x_clone.device,
                        dtype=torch.float32)
                    new_min = torch.tensor(np.percentile(
                        x_clone.reshape(-1).cpu(), (1 - pct) * 100),
                        device=x_clone.device,
                        dtype=torch.float32)   
                x_q = self.quantize(x_clone, new_max, new_min)
                score = lp_loss(x_clone, x_q, p=2, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                    zero_point = (- new_min / delta).round()

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q


class LogSqrt2Quantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, channel_wise: bool = False):
        super(LogSqrt2Quantizer, self).__init__()
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.inited = False
        self.channel_wise = channel_wise

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            self.delta = self.init_quantization_scale(x)
            self.inited = True

        # start quantization
        x_dequant = self.quantize(x, self.delta)
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor):
        delta = None
        x_clone = x.clone().detach()
        delta = x_clone.max()
        best_score = 1e+10
        for pct in [0.999, 0.9999, 0.99999]: #
            try:
                new_delta = torch.quantile(x_clone.reshape(-1), pct)
            except:
                new_delta = torch.tensor(np.percentile(
                    x_clone.reshape(-1).cpu(), pct * 100),
                    device=x_clone.device,
                    dtype=torch.float32)
            x_q = self.quantize(x_clone, new_delta)
            score = lp_loss(x_clone, x_q, p=2, reduction='all')
            if score < best_score:
                best_score = score
                delta = new_delta

        return delta

    def quantize(self, x, delta):      
        from math import sqrt
        x_int = torch.round(-1 * (x/delta).log2() * 2)
        mask = x_int >= self.n_levels
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        odd_mask = (x_quant%2) * (sqrt(2)-1) + 1
        x_float_q = 2**(-1 * torch.ceil(x_quant/2)) * odd_mask * delta
        x_float_q[mask] = 0
        
        return x_float_q
