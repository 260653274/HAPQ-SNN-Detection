from __future__ import annotations

import json
import os
import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from yolox.hapq.cost_model import HAPQBudget, HAPQCostModel
from yolox.hapq.integration import apply_candidate_to_model
from yolox.hapq.nas import CandidateEvaluation, ConstrainedNAS
from yolox.hapq.problem import HAPQSearchSpace


@dataclass
class HAPQPipelineResult:
    best_candidate: Dict
    history: List[Dict]
    stage_metrics: Dict[str, Dict[str, float]]
    output_json: str


class HAPQPipeline:
    def __init__(
        self,
        exp,
        run_baseline: Optional[Callable[[], Dict[str, float]]] = None,
        run_qat: Optional[Callable[[nn.Module], Dict[str, float]]] = None,
    ) -> None:
        self.exp = exp
        self.run_baseline_cb = run_baseline
        self.run_qat_cb = run_qat

    def _default_layer_names(self, model: nn.Module) -> List[str]:
        names = []
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                names.append(name)
        return names[: max(4, min(24, len(names)))]

    @staticmethod
    def save_hapq_checkpoint(
        path: str,
        model: nn.Module,
        best_candidate: Dict,
        mask_tensors: Dict[str, torch.Tensor],
        stage_metrics: Dict[str, Dict[str, float]],
        hapq_cfg: Dict,
    ) -> None:
        state = {
            "model": model.state_dict(),
            "best_candidate": best_candidate,
            "mask_tensors": {k: v.detach().cpu() for k, v in mask_tensors.items()},
            "stage_metrics": stage_metrics,
            "hapq_cfg": hapq_cfg,
        }
        torch.save(state, path)

    @staticmethod
    def load_hapq_checkpoint(path: str, map_location: str = "cpu") -> Dict:
        return torch.load(path, map_location=map_location)

    def run(self, model: nn.Module, output_dir: str) -> HAPQPipelineResult:
        os.makedirs(output_dir, exist_ok=True)
        layer_names = self._default_layer_names(model)
        if len(layer_names) == 0:
            raise RuntimeError("No searchable layers were found for HAPQ NAS.")

        cfg = self.exp.get_hapq_config() if hasattr(self.exp, "get_hapq_config") else {}
        budget = HAPQBudget(
            tau_lat=float(cfg.get("tau_lat", 2.5e7)),
            tau_eng=float(cfg.get("tau_eng", 1500.0)),
            tau_dsp=float(cfg.get("tau_dsp", 1600)),
            tau_bram=float(cfg.get("tau_bram", 1200)),
            tau_bw=float(cfg.get("tau_bw", 5e8)),
            tau_lut=float(cfg.get("tau_lut", 2e5)),
        )
        cost_model = HAPQCostModel(
            budget=budget,
            lambda_lat=float(cfg.get("lambda_lat", 1.0)),
            lambda_eng=float(cfg.get("lambda_eng", 1.0)),
            lambda_dsp=float(cfg.get("lambda_dsp", 1.0)),
            lambda_bram=float(cfg.get("lambda_bram", 1.0)),
            lambda_bw=float(cfg.get("lambda_bw", 1.0)),
            lambda_lut=float(cfg.get("lambda_lut", 1.0)),
        )
        search_space = HAPQSearchSpace(
            layer_names=layer_names,
            channel_choices=list(cfg.get("channel_choices", [64, 128, 256])),
            kernel_choices=list(cfg.get("kernel_choices", [1, 3, 5])),
            depth_choices=list(cfg.get("depth_choices", [1, 2, 3])),
            bit_choices_w=list(cfg.get("bit_choices_w", [4, 6, 8, 12])),
            bit_choices_u=list(cfg.get("bit_choices_u", [6, 8, 12, 16])),
            block_size=int(cfg.get("block_size", 8)),
            leak_shift_choices=list(cfg.get("leak_shift_choices", [2, 3, 4])),
        )

        stage_metrics: Dict[str, Dict[str, float]] = {}
        if self.run_baseline_cb is not None:
            stage_metrics["baseline"] = self.run_baseline_cb()
        else:
            stage_metrics["baseline"] = {"status": 1.0}

        # Hardware calibration placeholder can be swapped for real board measurements.
        stage_metrics["calibration"] = {"status": 1.0}

        nas = ConstrainedNAS(
            search_space=search_space,
            cost_model=cost_model,
            perf_evaluator=None,
            seed=int(cfg.get("seed", 0)),
        )
        best, history = nas.search(
            base_model=model,
            num_iters=int(cfg.get("nas_iters", 20)),
            population_size=int(cfg.get("nas_population", 8)),
            top_k=int(cfg.get("nas_topk", 3)),
            mutation_rate=float(cfg.get("nas_mutation", 0.25)),
        )

        apply_info = apply_candidate_to_model(
            model,
            best,
            default_block_size=int(cfg.get("block_size", 8)),
            apply_mode=cfg.get("stage", "full"),
        )
        mask_path = os.path.join(output_dir, "hapq_mask.pt")
        torch.save({k: v.detach().cpu() for k, v in apply_info["masks"].items()}, mask_path)
        stage_metrics["pruning_quantization"] = {
            "keep_ratio": float(apply_info["keep_ratio"]),
            "mask_layers": float(len(apply_info["masks"])),
        }

        if self.run_qat_cb is not None:
            stage_metrics["qat"] = self.run_qat_cb(model)
        else:
            stage_metrics["qat"] = {"status": 1.0}

        hapq_ckpt_path = os.path.join(output_dir, "hapq_best_ckpt.pth")
        self.save_hapq_checkpoint(
            path=hapq_ckpt_path,
            model=model,
            best_candidate=best.to_dict(),
            mask_tensors=apply_info["masks"],
            stage_metrics=stage_metrics,
            hapq_cfg=cfg,
        )

        # Create effective candidate for export based on apply_mode
        effective_candidate = copy.deepcopy(best)
        apply_mode = cfg.get("stage", "full")
        print(f"DEBUG: apply_mode='{apply_mode}'")
        
        if apply_mode == "prune_only":
            for layer in effective_candidate.layers:
                layer.b_w = 32
                layer.b_u = 32
        elif apply_mode == "quant_w":
            for layer in effective_candidate.layers:
                layer.b_u = 32
                # Reset active_blocks to total_blocks (no pruning)
                layer.active_blocks = layer.total_blocks
        elif apply_mode == "quant_wu":
            print(f"DEBUG: Processing quant_wu. Layers: {len(effective_candidate.layers)}")
            for i, layer in enumerate(effective_candidate.layers):
                old_active = layer.active_blocks
                layer.active_blocks = layer.total_blocks
                print(f"DEBUG: Layer {i} {layer.name}: active {old_active} -> {layer.active_blocks} (total {layer.total_blocks})")
        elif apply_mode == "baseline":
            for layer in effective_candidate.layers:
                layer.b_w = 32
                layer.b_u = 32
                layer.active_blocks = layer.total_blocks

        summary = {
            "best_candidate": effective_candidate.to_dict(),
            "hapq_cfg": cfg,
            "history": [
                {
                    "objective": item.objective,
                    "perf_loss": item.perf_loss,
                    "resources": item.resources,
                    "generation": item.generation,
                    "penalty_lat": item.penalty_lat,
                    "penalty_eng": item.penalty_eng,
                    "penalty_dsp": item.penalty_dsp,
                    "penalty_bram": item.penalty_bram,
                    "penalty_bw": item.penalty_bw,
                    "penalty_lut": item.penalty_lut,
                    "candidate": item.candidate.to_dict(),
                }
                for item in history
            ],
            "stage_metrics": stage_metrics,
            "artifacts": {
                "mask_path": mask_path,
                "hapq_ckpt": hapq_ckpt_path,
            },
        }
        output_json = os.path.join(output_dir, "hapq_summary.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return HAPQPipelineResult(
            best_candidate=summary["best_candidate"],
            history=summary["history"],
            stage_metrics=stage_metrics,
            output_json=output_json,
        )
