from __future__ import annotations

import dataclasses
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class HAPQBudget:
    tau_lat: float
    tau_eng: float
    tau_dsp: float
    tau_bram: float
    tau_bw: float = float("inf")
    tau_lut: float = float("inf")


@dataclass
class HAPQLayerChoice:
    name: str
    channels: int
    kernel: int
    depth: int
    active_blocks: int
    total_blocks: int
    b_w: int
    b_u: int
    leak_shift_n: int
    activity: float

    @property
    def keep_ratio(self) -> float:
        if self.total_blocks <= 0:
            return 1.0
        return float(self.active_blocks) / float(self.total_blocks)

    def validate(self) -> None:
        if self.active_blocks < 1 or self.active_blocks > self.total_blocks:
            raise ValueError(f"Invalid block selection for layer {self.name}.")
        if self.b_w <= 0 or self.b_u <= 0:
            raise ValueError(f"Invalid bit width in layer {self.name}.")
        if self.leak_shift_n < 0:
            raise ValueError(f"Invalid leak shift n in layer {self.name}.")


@dataclass
class HAPQCandidate:
    layers: List[HAPQLayerChoice] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, float] = field(default_factory=dict)

    def validate(self) -> None:
        for layer in self.layers:
            layer.validate()

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class HAPQSearchSpace:
    """Search space over architecture a, structured mask m, and quantization q."""

    def __init__(
        self,
        layer_names: List[str],
        channel_choices: List[int],
        kernel_choices: List[int],
        depth_choices: List[int],
        bit_choices_w: List[int],
        bit_choices_u: List[int],
        block_size: int = 8,
        leak_shift_choices: List[int] | None = None,
    ) -> None:
        self.layer_names = layer_names
        self.channel_choices = channel_choices
        self.kernel_choices = kernel_choices
        self.depth_choices = depth_choices
        self.bit_choices_w = bit_choices_w
        self.bit_choices_u = bit_choices_u
        self.block_size = block_size
        self.leak_shift_choices = leak_shift_choices if leak_shift_choices is not None else [2, 3, 4]

    def sample_candidate(self, rng: random.Random) -> HAPQCandidate:
        layers: List[HAPQLayerChoice] = []
        for name in self.layer_names:
            channels = rng.choice(self.channel_choices)
            total_blocks = max(1, channels // max(1, self.block_size))
            active_blocks = rng.randint(max(1, total_blocks // 4), total_blocks)
            layers.append(
                HAPQLayerChoice(
                    name=name,
                    channels=channels,
                    kernel=rng.choice(self.kernel_choices),
                    depth=rng.choice(self.depth_choices),
                    active_blocks=active_blocks,
                    total_blocks=total_blocks,
                    b_w=rng.choice(self.bit_choices_w),
                    b_u=rng.choice(self.bit_choices_u),
                    leak_shift_n=rng.choice(self.leak_shift_choices),
                    activity=rng.uniform(0.05, 0.8),
                )
            )
        candidate = HAPQCandidate(layers=layers)
        candidate.validate()
        return candidate

    def mutate(self, parent: HAPQCandidate, rng: random.Random, mutation_rate: float = 0.25) -> HAPQCandidate:
        layers: List[HAPQLayerChoice] = []
        for layer in parent.layers:
            data = dataclasses.asdict(layer)
            if rng.random() < mutation_rate:
                data["channels"] = rng.choice(self.channel_choices)
                data["total_blocks"] = max(1, data["channels"] // max(1, self.block_size))
                data["active_blocks"] = min(data["active_blocks"], data["total_blocks"])
            if rng.random() < mutation_rate:
                data["kernel"] = rng.choice(self.kernel_choices)
            if rng.random() < mutation_rate:
                data["depth"] = rng.choice(self.depth_choices)
            if rng.random() < mutation_rate:
                data["b_w"] = rng.choice(self.bit_choices_w)
            if rng.random() < mutation_rate:
                data["b_u"] = rng.choice(self.bit_choices_u)
            if rng.random() < mutation_rate:
                data["leak_shift_n"] = rng.choice(self.leak_shift_choices)
            if rng.random() < mutation_rate:
                data["active_blocks"] = rng.randint(max(1, data["total_blocks"] // 4), data["total_blocks"])
            if rng.random() < mutation_rate:
                data["activity"] = rng.uniform(0.05, 0.8)
            layers.append(HAPQLayerChoice(**data))
        child = HAPQCandidate(layers=layers)
        child.validate()
        return child
