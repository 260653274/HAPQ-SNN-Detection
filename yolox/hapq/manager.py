from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict

from .quantization.layers import QATConv2d
from .pruning import StructuredPruningConfig, apply_structured_pruning


class HAPQManager:
    """
    Manages the application of pruning masks and quantization simulation (QAT) 
    during training/finetuning.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.masks: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def register_pruning_hooks(self, masks: Dict[str, torch.Tensor]):
        """
        Registers forward pre-hooks to enforce sparsity mask on weights before each forward pass.
        This ensures that even if optimizer updates the weights, the pruned weights remain zero
        during the forward pass.
        """
        self.masks = masks
        
        def get_hook(mask_tensor):
            def hook(module, input):
                # Apply mask to weight in-place before forward
                with torch.no_grad():
                    module.weight.data.mul_(mask_tensor.to(module.weight.device))
            return hook

        for name, module in self.model.named_modules():
            if name in self.masks:
                # Resize mask to match weight tensor dimensions
                mask_tensor = self.masks[name]
                if mask_tensor.dim() == 1: # Channel mask
                     mask_tensor = mask_tensor.view(-1, 1, 1, 1) # [C_out, 1, 1, 1]
                
                h = module.register_forward_pre_hook(get_hook(mask_tensor))
                self.hooks.append(h)

    def convert_to_qat_model(self, bit_width_map: Dict[str, int], default_bw: int = 8):
        """
        Replaces standard nn.Conv2d layers with QATConv2d layers that perform 
        fake quantization on weights during forward pass.
        """
        replacements = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and not isinstance(module, QATConv2d):
                replacements.append((name, module))
        
        for name, module in replacements:
            bw = bit_width_map.get(name, default_bw)
            qat_layer = QATConv2d.from_float(module, bit_width=bw)
            
            # Replace in parent module
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = self.model.get_submodule(parent_name)
            else:
                parent = self.model
                child_name = name
                
            setattr(parent, child_name, qat_layer)

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
