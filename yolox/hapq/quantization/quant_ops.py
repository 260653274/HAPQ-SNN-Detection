from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class QuantConfig:
    b_w: int = 8
    b_u: int = 8
    eps: float = 1e-8


def quantize_tensor_symmetric(x: torch.Tensor, bit_width: int, eps: float = 1e-8) -> torch.Tensor:
    qmax = float(2 ** (bit_width - 1) - 1)
    max_abs = torch.max(torch.abs(x)).detach()
    scale = torch.clamp(max_abs / qmax, min=eps)
    x_int = torch.round(x / scale).clamp(-qmax - 1, qmax)
    x_q = x_int * scale
    # Straight-through estimator for QAT
    return x + (x_q - x).detach()


def apply_weight_quantization(model: nn.Module, bit_width_map: Dict[str, int], default_bw: int = 8) -> None:
    for name, module in model.named_modules():
        if not hasattr(module, "weight") or module.weight is None:
            continue
        if not isinstance(module.weight, torch.Tensor):
            continue
        bw = bit_width_map.get(name, default_bw)
        module.weight.data = quantize_tensor_symmetric(module.weight.data, bit_width=bw)


def quantize_membrane_update(
    u_prev: torch.Tensor,
    input_q: torch.Tensor,
    spike_prev: torch.Tensor,
    theta_q: torch.Tensor | float,
    leak_shift_n: int,
    bit_width_u: int,
) -> torch.Tensor:
    # beta = 1 - 2^{-n} implemented with shift-friendly decay
    leak_term = torch.floor(u_prev / float(2 ** leak_shift_n))
    u_next = u_prev - leak_term + input_q - theta_q * spike_prev
    return quantize_tensor_symmetric(u_next, bit_width=bit_width_u)
