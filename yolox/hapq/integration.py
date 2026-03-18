from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any
from spikingjelly.activation_based import neuron

from .pruning import StructuredPruningConfig, apply_structured_pruning
from .problem import HAPQCandidate
from .quantization import apply_weight_quantization
from .quantization.neuron import QuantizedParametricLIFNode
from .manager import HAPQManager
from .quantization.layers import QATConv2d


def apply_membrane_quantization(
    model: nn.Module,
    bit_width_map: Dict[str, int],
    leak_shift_map: Dict[str, int],
    default_bw: int = 8,
    default_leak_shift: int = 0,
) -> None:
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, neuron.ParametricLIFNode) and not isinstance(module, QuantizedParametricLIFNode):
             replacements.append((name, module))
             
    for name, module in replacements:
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name
            
        bw = bit_width_map.get(name, default_bw)
        leak_shift = leak_shift_map.get(name, default_leak_shift)
        
        new_node = QuantizedParametricLIFNode(
            init_tau=module.init_tau,
            decay_input=module.decay_input,
            v_threshold=module.v_threshold,
            v_reset=module.v_reset,
            surrogate_function=module.surrogate_function,
            detach_reset=module.detach_reset,
            step_mode=module.step_mode,
            backend=module.backend,
            store_v_seq=module.store_v_seq,
            b_u=bw,
            leak_shift_n=leak_shift
        )
        if hasattr(module, 'w'):
            new_node.w.data = module.w.data.clone()
            
        setattr(parent, child_name, new_node)


def apply_candidate_to_model(
    model: nn.Module,
    candidate: HAPQCandidate,
    default_block_size: int = 8,
    apply_mode: str = "full",
) -> Dict[str, Any]:
    # Build per-layer quant map and global pruning ratio.
    bit_width_w_map = {layer.name: layer.b_w for layer in candidate.layers}
    bit_width_u_map = {layer.name: layer.b_u for layer in candidate.layers}
    leak_shift_map = {layer.name: getattr(layer, "leak_shift_n", 0) for layer in candidate.layers}
    
    keep_ratios = [layer.keep_ratio for layer in candidate.layers]
    keep_ratio = sum(keep_ratios) / max(1, len(keep_ratios))
    
    masks = {}
    manager = HAPQManager(model)

    # 1. Apply Weight Quantization (Convert to QAT layers)
    if apply_mode in ["full", "quant_w", "quant_wu"]:
        # Instead of static quantization, convert to QAT layers
        # apply_weight_quantization(model, bit_width_map=bit_width_w_map, default_bw=8)
        manager.convert_to_qat_model(bit_width_w_map, default_bw=8)

    # 2. Apply Membrane Quantization (Replace LIF nodes)
    if apply_mode in ["full", "quant_wu"]:
        apply_membrane_quantization(
            model,
            bit_width_map=bit_width_u_map,
            leak_shift_map=leak_shift_map,
            default_bw=8,
            default_leak_shift=0
        )

    # 3. Apply Structured Pruning
    if apply_mode in ["full", "prune_only"]:
        masks = apply_structured_pruning(
            model,
            StructuredPruningConfig(block_size=default_block_size, keep_ratio=keep_ratio),
        )
        # Register hooks to enforce sparsity during training
        manager.register_pruning_hooks(masks)

    return {"keep_ratio": keep_ratio, "masks": masks, "manager": manager}
