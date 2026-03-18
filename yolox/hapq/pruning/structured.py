from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class StructuredPruningConfig:
    block_size: int = 8
    keep_ratio: float = 0.8
    min_keep_blocks: int = 1


def collect_conv2d_layers(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    layers: List[Tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.weight is not None:
            layers.append((name, module))
    return layers


def compute_taylor_scores(layer: nn.Conv2d) -> torch.Tensor:
    weight = layer.weight.detach()
    grad = layer.weight.grad
    if grad is None:
        # fallback when no gradient available yet
        grad = torch.ones_like(weight)
    # group importance by output channel
    score = torch.sum(torch.abs(weight * grad), dim=(1, 2, 3))
    return score


def _block_mask_from_scores(scores: torch.Tensor, block_size: int, keep_ratio: float, min_keep_blocks: int) -> torch.Tensor:
    out_channels = scores.numel()
    block_size = max(1, block_size)
    block_count = (out_channels + block_size - 1) // block_size
    block_scores = []
    for b in range(block_count):
        start = b * block_size
        end = min((b + 1) * block_size, out_channels)
        block_scores.append(torch.mean(scores[start:end]))
    block_scores_tensor = torch.stack(block_scores)
    keep_blocks = max(min_keep_blocks, int(round(keep_ratio * block_count)))
    keep_blocks = min(block_count, keep_blocks)
    _, top_idx = torch.topk(block_scores_tensor, k=keep_blocks, largest=True)
    block_mask = torch.zeros(block_count, device=scores.device, dtype=scores.dtype)
    block_mask[top_idx] = 1.0
    channel_mask = torch.zeros(out_channels, device=scores.device, dtype=scores.dtype)
    for b in range(block_count):
        if block_mask[b] > 0:
            start = b * block_size
            end = min((b + 1) * block_size, out_channels)
            channel_mask[start:end] = 1.0
    return channel_mask


def apply_structured_pruning(
    model: nn.Module, config: StructuredPruningConfig
) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    for name, layer in collect_conv2d_layers(model):
        scores = compute_taylor_scores(layer)
        channel_mask = _block_mask_from_scores(
            scores=scores,
            block_size=config.block_size,
            keep_ratio=config.keep_ratio,
            min_keep_blocks=config.min_keep_blocks,
        )
        weight_mask = channel_mask.view(-1, 1, 1, 1)
        layer.weight.data.mul_(weight_mask)
        masks[name] = channel_mask.detach().cpu()
    return masks
