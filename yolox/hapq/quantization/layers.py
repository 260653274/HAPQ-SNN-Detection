from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant_ops import quantize_tensor_symmetric


class QATConv2d(nn.Conv2d):
    """
    Conv2d with fake quantization for Quantization-Aware Training (QAT).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | str | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        bit_width: int = 8,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype
        )
        self.bit_width = bit_width

    @classmethod
    def from_float(cls, mod: nn.Conv2d, bit_width: int = 8) -> QATConv2d:
        """
        Create a QATConv2d from a float nn.Conv2d module.
        """
        qat_conv = cls(
            in_channels=mod.in_channels,
            out_channels=mod.out_channels,
            kernel_size=mod.kernel_size,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
            bias=(mod.bias is not None),
            padding_mode=mod.padding_mode,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
            bit_width=bit_width,
        )
        with torch.no_grad():
            qat_conv.weight.copy_(mod.weight)
            if mod.bias is not None:
                qat_conv.bias.copy_(mod.bias)
        return qat_conv

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Fake Quantization (Weight)
        # Using straight-through estimator via detach logic inside quantize_tensor_symmetric
        weight_q = quantize_tensor_symmetric(self.weight, self.bit_width)
        
        return self._conv_forward(input, weight_q, self.bias)
