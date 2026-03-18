from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List


def _hinge(value: float) -> float:
    return max(0.0, value)


@dataclass
class HAPQBudget:
    tau_lat: float
    tau_eng: float
    tau_dsp: float
    tau_bram: float
    tau_bw: float = float("inf")
    tau_lut: float = float("inf")


@dataclass
class LayerCostSpec:
    name: str
    p_req: float
    dense_synops: float
    activity: float
    mask_keep_ratio: float
    b_w: int
    b_u: int
    state_neurons: int
    timesteps: int
    bram_width: int = 36
    bram_depth: int = 1024


@dataclass
class ResourceEstimate:
    lat: float
    eng: float
    dsp: float
    bram: float
    bw: float
    lut: float
    synops: float
    per_layer: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class HAPQObjective:
    det_loss: float
    total_loss: float
    penalty_lat: float
    penalty_eng: float
    penalty_dsp: float
    penalty_bram: float
    penalty_bw: float
    penalty_lut: float
    resources: ResourceEstimate


class HAPQCostModel:
    """Analytic estimator for constrained HAPQ objective."""

    def __init__(
        self,
        budget: HAPQBudget,
        lambda_lat: float = 1.0,
        lambda_eng: float = 1.0,
        lambda_dsp: float = 1.0,
        lambda_bram: float = 1.0,
        lambda_bw: float = 1.0,
        lambda_lut: float = 1.0,
    ) -> None:
        self.budget = budget
        self.lambda_lat = lambda_lat
        self.lambda_eng = lambda_eng
        self.lambda_dsp = lambda_dsp
        self.lambda_bram = lambda_bram
        self.lambda_bw = lambda_bw
        self.lambda_lut = lambda_lut

    @staticmethod
    def dsp_packing_factor(bit_width: int) -> float:
        # Conservative default table for DSP48-like packing.
        table = {
            16: 1.0,
            12: 1.0,
            8: 2.0,
            6: 3.0,
            4: 4.0,
        }
        if bit_width in table:
            return table[bit_width]
        if bit_width <= 4:
            return 4.0
        if bit_width <= 8:
            return 2.0
        return 1.0

    @staticmethod
    def bram_width_penalty(width: int) -> float:
        if width <= 18:
            return 1.0
        if width <= 36:
            return 1.1
        return 1.25

    @staticmethod
    def estimate_layer_dsp(spec: LayerCostSpec) -> float:
        packed = HAPQCostModel.dsp_packing_factor(spec.b_w)
        return math.ceil(max(1.0, spec.p_req * spec.mask_keep_ratio) / packed)

    @staticmethod
    def estimate_layer_bram(spec: LayerCostSpec) -> float:
        width = max(1, spec.b_u)
        depth = max(1, spec.state_neurons * max(1, spec.timesteps))
        bits = width * depth
        capacity = max(1, spec.bram_width * spec.bram_depth)
        return math.ceil(bits / capacity) * HAPQCostModel.bram_width_penalty(width)

    @staticmethod
    def estimate_layer_synops(spec: LayerCostSpec) -> float:
        return spec.timesteps * spec.activity * spec.dense_synops * spec.mask_keep_ratio

    @staticmethod
    def estimate_layer_bw(spec: LayerCostSpec) -> float:
        # approximate bytes moved per step
        return spec.state_neurons * (spec.b_u / 8.0) * spec.timesteps

    @staticmethod
    def estimate_layer_lut(spec: LayerCostSpec) -> float:
        # lightweight proxy from quantized datapath complexity
        return (spec.b_w + spec.b_u) * max(1.0, spec.p_req * spec.mask_keep_ratio) * 0.25

    def estimate(self, layer_specs: List[LayerCostSpec]) -> ResourceEstimate:
        per_layer: Dict[str, Dict[str, float]] = {}
        total_dsp = 0.0
        total_bram = 0.0
        total_synops = 0.0
        total_bw = 0.0
        total_lut = 0.0
        for spec in layer_specs:
            dsp = self.estimate_layer_dsp(spec)
            bram = self.estimate_layer_bram(spec)
            synops = self.estimate_layer_synops(spec)
            bw = self.estimate_layer_bw(spec)
            lut = self.estimate_layer_lut(spec)
            total_dsp += dsp
            total_bram += bram
            total_synops += synops
            total_bw += bw
            total_lut += lut
            per_layer[spec.name] = {
                "dsp": float(dsp),
                "bram": float(bram),
                "synops": float(synops),
                "bw": float(bw),
                "lut": float(lut),
            }

        # Proxy latency/energy from resource and operation mix.
        lat = total_synops / max(1.0, total_dsp)
        eng = total_synops * 1e-6 + total_bram * 0.01 + total_bw * 1e-4
        return ResourceEstimate(
            lat=lat,
            eng=eng,
            dsp=total_dsp,
            bram=total_bram,
            bw=total_bw,
            lut=total_lut,
            synops=total_synops,
            per_layer=per_layer,
        )

    def objective(self, det_loss: float, layer_specs: List[LayerCostSpec]) -> HAPQObjective:
        resources = self.estimate(layer_specs)
        penalty_lat = self.lambda_lat * _hinge(resources.lat - self.budget.tau_lat)
        penalty_eng = self.lambda_eng * _hinge(resources.eng - self.budget.tau_eng)
        penalty_dsp = self.lambda_dsp * _hinge(resources.dsp - self.budget.tau_dsp)
        penalty_bram = self.lambda_bram * _hinge(resources.bram - self.budget.tau_bram)
        penalty_bw = self.lambda_bw * _hinge(resources.bw - self.budget.tau_bw)
        penalty_lut = self.lambda_lut * _hinge(resources.lut - self.budget.tau_lut)
        total = det_loss + penalty_lat + penalty_eng + penalty_dsp + penalty_bram + penalty_bw + penalty_lut
        return HAPQObjective(
            det_loss=det_loss,
            total_loss=total,
            penalty_lat=penalty_lat,
            penalty_eng=penalty_eng,
            penalty_dsp=penalty_dsp,
            penalty_bram=penalty_bram,
            penalty_bw=penalty_bw,
            penalty_lut=penalty_lut,
            resources=resources,
        )
