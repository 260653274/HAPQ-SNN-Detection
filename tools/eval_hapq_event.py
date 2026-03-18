#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import json
import os
import random
import warnings
from typing import Dict

from loguru import logger
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from yolox.exp import get_exp
from yolox.hapq.pipeline import HAPQPipeline
from yolox.hapq.integration import apply_candidate_to_model
from yolox.hapq.problem import HAPQCandidate, HAPQLayerChoice
from yolox.utils import configure_module, get_model_info


def make_parser():
    parser = argparse.ArgumentParser("EAS-SNN HAPQ eval parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint for eval")
    parser.add_argument("--hapq-json", default=None, type=str, help="hapq summary json path")
    parser.add_argument("--hapq-ckpt", default=None, type=str, help="hapq pipeline checkpoint path")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    parser.add_argument("--batch-size", default=8, type=int, help="eval batch size")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def candidate_from_json(path: str) -> HAPQCandidate:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    candidate_data = data["best_candidate"] if "best_candidate" in data else data
    layers = [HAPQLayerChoice(**layer) for layer in candidate_data["layers"]]
    return HAPQCandidate(layers=layers, score=candidate_data.get("score", 0.0), metadata=candidate_data.get("metadata", {}))


def candidate_from_dict(candidate_data: Dict) -> HAPQCandidate:
    layers = [HAPQLayerChoice(**layer) for layer in candidate_data["layers"]]
    return HAPQCandidate(layers=layers, score=candidate_data.get("score", 0.0), metadata=candidate_data.get("metadata", {}))


def apply_mask_tensors(model: nn.Module, mask_tensors: Dict[str, torch.Tensor]) -> None:
    modules = dict(model.named_modules())
    for name, channel_mask in mask_tensors.items():
        module = modules.get(name)
        if module is None or not hasattr(module, "weight"):
            continue
        weight = module.weight.data
        if channel_mask.device != weight.device:
            channel_mask = channel_mask.to(weight.device)
        weight_mask = channel_mask.view(-1, 1, 1, 1).to(weight.dtype)
        weight.mul_(weight_mask)


@logger.catch
def main(exp, args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed eval. CUDNN deterministic is enabled.")
    cudnn.benchmark = True

    model = exp.get_model()
    logger.info("Model Summary: {}", get_model_info(model, exp.test_size, in_dim=exp.in_dim))
    hapq_payload = None
    if args.hapq_ckpt is not None and os.path.exists(args.hapq_ckpt):
        hapq_payload = HAPQPipeline.load_hapq_checkpoint(args.hapq_ckpt, map_location="cpu")
        logger.info("Loaded HAPQ checkpoint from {}", args.hapq_ckpt)

    candidate = None
    if hapq_payload is not None and "best_candidate" in hapq_payload:
        candidate = candidate_from_dict(hapq_payload["best_candidate"])
    elif args.hapq_json is not None:
        candidate = candidate_from_json(args.hapq_json)
    else:
        raise ValueError("Either --hapq-ckpt or --hapq-json must be provided for HAPQ eval.")

    apply_candidate_to_model(model, candidate, default_block_size=exp.hapq_block_size)
    if hapq_payload is not None and "mask_tensors" in hapq_payload:
        apply_mask_tensors(model, hapq_payload["mask_tensors"])

    if args.ckpt is not None:
        ckpt_file = args.ckpt
    elif hapq_payload is not None and "model" in hapq_payload:
        ckpt_file = args.hapq_ckpt
    else:
        ckpt_file = os.path.join(exp.output_dir, args.experiment_name, "best_ckpt.pth")
    if ckpt_file is not None and os.path.exists(ckpt_file):
        if hapq_payload is not None and ckpt_file == args.hapq_ckpt and "model" in hapq_payload:
            state = hapq_payload["model"]
        else:
            ckpt = torch.load(ckpt_file, map_location="cpu")
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint weights from {}", ckpt_file)
    else:
        logger.warning("Checkpoint not found: {}. Evaluate with current model weights.", ckpt_file)

    is_distributed = False
    evaluator = exp.get_evaluator(args.batch_size, is_distributed)
    model.cuda()
    model.eval()
    *_, summary = evaluator.evaluate(model, is_distributed, half=False, trt_file=None, decoder=None, test_size=exp.test_size)
    logger.info("\n{}", summary)


if __name__ == "__main__":
    configure_module()
    parser = make_parser()
    args = parser.parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.enable_hapq = True
    exp.merge(args.opts)
    if not args.experiment_name:
        args.experiment_name = f"{exp.exp_name}_hapq"
    main(exp, args)
