from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch.nn as nn

from yolox.hapq.cost_model import HAPQCostModel, LayerCostSpec
from yolox.hapq.problem import HAPQCandidate, HAPQSearchSpace


@dataclass
class CandidateEvaluation:
    candidate: HAPQCandidate
    perf_loss: float
    objective: float
    resources: dict
    generation: int = -1
    penalty_lat: float = 0.0
    penalty_eng: float = 0.0
    penalty_dsp: float = 0.0
    penalty_bram: float = 0.0
    penalty_bw: float = 0.0
    penalty_lut: float = 0.0


class ConstrainedNAS:
    """Random-evolution NAS for (a, m, q) under hardware penalties."""

    def __init__(
        self,
        search_space: HAPQSearchSpace,
        cost_model: HAPQCostModel,
        perf_evaluator: Optional[Callable[[nn.Module], float]] = None,
        seed: int = 0,
    ) -> None:
        self.search_space = search_space
        self.cost_model = cost_model
        self.perf_evaluator = perf_evaluator
        self.rng = random.Random(seed)

    def _layer_specs_from_candidate(self, candidate: HAPQCandidate) -> List[LayerCostSpec]:
        specs: List[LayerCostSpec] = []
        for layer in candidate.layers:
            # p_req and dense_synops are proxies from architecture choices.
            p_req = float(layer.channels * layer.kernel * layer.kernel * layer.depth)
            dense_synops = p_req * 32.0
            specs.append(
                LayerCostSpec(
                    name=layer.name,
                    p_req=p_req,
                    dense_synops=dense_synops,
                    activity=layer.activity,
                    mask_keep_ratio=layer.keep_ratio,
                    b_w=layer.b_w,
                    b_u=layer.b_u,
                    state_neurons=layer.channels * 4,
                    timesteps=4,
                )
            )
        return specs

    def _estimate_perf_loss(self, model: nn.Module, candidate: HAPQCandidate) -> float:
        if self.perf_evaluator is not None:
            return float(self.perf_evaluator(model))
        # Heuristic proxy if no evaluator is provided.
        avg_keep = sum(layer.keep_ratio for layer in candidate.layers) / max(1, len(candidate.layers))
        avg_bw = sum(layer.b_w for layer in candidate.layers) / max(1, len(candidate.layers))
        avg_bu = sum(layer.b_u for layer in candidate.layers) / max(1, len(candidate.layers))
        quant_penalty = (16.0 - avg_bw) * 0.01 + (16.0 - avg_bu) * 0.008
        sparsity_penalty = (1.0 - avg_keep) * 0.5
        return max(0.0, quant_penalty + sparsity_penalty)

    def evaluate(self, model: nn.Module, candidate: HAPQCandidate) -> CandidateEvaluation:
        perf_loss = self._estimate_perf_loss(model, candidate)
        specs = self._layer_specs_from_candidate(candidate)
        objective = self.cost_model.objective(det_loss=perf_loss, layer_specs=specs)
        candidate.score = -objective.total_loss
        candidate.metadata = {
            "perf_loss": perf_loss,
            "objective": objective.total_loss,
            "dsp": objective.resources.dsp,
            "bram": objective.resources.bram,
            "lat": objective.resources.lat,
            "eng": objective.resources.eng,
            "penalty_dsp": objective.penalty_dsp,
            "penalty_bram": objective.penalty_bram,
        }
        return CandidateEvaluation(
            candidate=candidate,
            perf_loss=perf_loss,
            objective=objective.total_loss,
            resources={
                "dsp": objective.resources.dsp,
                "bram": objective.resources.bram,
                "lat": objective.resources.lat,
                "eng": objective.resources.eng,
                "synops": objective.resources.synops,
                "bw": objective.resources.bw,
                "lut": objective.resources.lut,
            },
            penalty_lat=objective.penalty_lat,
            penalty_eng=objective.penalty_eng,
            penalty_dsp=objective.penalty_dsp,
            penalty_bram=objective.penalty_bram,
            penalty_bw=objective.penalty_bw,
            penalty_lut=objective.penalty_lut,
        )

    def search(
        self,
        base_model: nn.Module,
        num_iters: int = 20,
        population_size: int = 8,
        top_k: int = 3,
        mutation_rate: float = 0.25,
    ) -> Tuple[HAPQCandidate, List[CandidateEvaluation]]:
        population = [self.search_space.sample_candidate(self.rng) for _ in range(population_size)]
        history: List[CandidateEvaluation] = []

        from tqdm import tqdm
        print(f"Starting NAS search for {num_iters} generations...")

        for generation in tqdm(range(num_iters), desc="NAS Search"):
            evaluated = [self.evaluate(base_model, cand) for cand in population]
            for item in evaluated:
                item.generation = generation
            evaluated.sort(key=lambda item: item.objective)
            history.extend(evaluated)
            elites = evaluated[: max(1, min(top_k, len(evaluated)))]

            # Regenerate population by mutating elites.
            next_population: List[HAPQCandidate] = [copy.deepcopy(elite.candidate) for elite in elites]
            while len(next_population) < population_size:
                parent = self.rng.choice(elites).candidate
                child = self.search_space.mutate(parent=parent, rng=self.rng, mutation_rate=mutation_rate)
                next_population.append(child)
            population = next_population

        history.sort(key=lambda item: item.objective)
        best = history[0].candidate
        print("NAS search completed.")
        return best, history
