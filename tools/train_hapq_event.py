#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import copy
import csv
import os
import random
import warnings
from argparse import Namespace
from typing import Dict, List

from loguru import logger
import torch
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from yolox.data import DataPrefetcher
from yolox.exp import check_exp_value, get_exp, BaseExp
from yolox.hapq.pipeline import HAPQPipeline
from yolox.utils import configure_module


def make_parser():
    parser = argparse.ArgumentParser("EAS-SNN HAPQ train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("-c", "--ckpt", default="ckpt/best_ckpt.pth", type=str, help="checkpoint for warm start")
    parser.add_argument("--seed", default=None, type=int, help="random seed")
    parser.add_argument("--skip-baseline", action="store_true", default=False, help="skip baseline training")
    parser.add_argument("--skip-qat", action="store_true", default=False, help="skip QAT stage")
    parser.add_argument("--qat-epochs", type=int, default=10, help="number of lightweight QAT epochs")
    parser.add_argument("--qat-lr", type=float, default=2e-5, help="QAT learning rate override")
    parser.add_argument("--qat-grad-clip", type=float, default=1.0, help="gradient norm clip for QAT")
    parser.add_argument("--qat-eval-interval", type=int, default=1, help="evaluate every N QAT epochs")
    parser.add_argument("--freeze-bn-epoch", type=int, default=-1, help="epoch to start freezing BN (default: -1, never)")
    parser.add_argument(
        "--search-log-csv",
        default=None,
        type=str,
        help="optional csv path to dump per-generation NAS best trajectory",
    )
    parser.add_argument("--output-json", default=None, type=str, help="path of hapq summary json")
    parser.add_argument("--hapq-force-bitwidth-w", type=int, default=None, help="Force weight bitwidth for all layers")
    parser.add_argument("--hapq-force-bitwidth-u", type=int, default=None, help="Force membrane bitwidth for all layers")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def _is_spike_enabled(flag) -> bool:
    if isinstance(flag, bool):
        return flag
    if isinstance(flag, str):
        return flag.lower() not in ("false", "0", "", "none")
    return bool(flag)


def maybe_load_ckpt(model: torch.nn.Module, ckpt_path: str | None, device: str = "cpu") -> None:
    if ckpt_path is None:
        return
    logger.info("Loading warm-start checkpoint from {}", ckpt_path)
    # PyTorch 2.6+ changed default to weights_only=True which may block numpy scalars.
    # We trust our own checkpoint file, so we disable this check or allow unsafe globals.
    # Trying weights_only=False for backward compatibility with older checkpoints containing numpy types.
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
         # Fallback for older torch versions where weights_only argument doesn't exist
        state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 0:
        logger.warning("Missing {} keys while loading HAPQ warm-start ckpt.", len(missing))
    if len(unexpected) > 0:
        logger.warning("Unexpected {} keys while loading HAPQ warm-start ckpt.", len(unexpected))


def _parse_eval_ap(eval_result):
    ap50_95 = 0.0
    ap50 = 0.0
    if isinstance(eval_result, tuple):
        if len(eval_result) >= 1 and isinstance(eval_result[0], tuple):
            first = eval_result[0]
            if len(first) >= 2:
                ap50_95 = float(first[0])
                ap50 = float(first[1])
        elif len(eval_result) >= 1 and isinstance(eval_result[0], (float, int)):
            ap50_95 = float(eval_result[0])
            if len(eval_result) >= 2:
                 ap50 = float(eval_result[1])
    return ap50_95, ap50


def _build_qat_optimizer(model: torch.nn.Module, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters found for QAT.")
    return torch.optim.Adam(params, lr=lr)


def export_search_log(history: List[Dict], out_csv: str) -> None:
    by_generation: Dict[int, Dict] = {}
    for item in history:
        generation = int(item.get("generation", -1))
        if generation < 0:
            continue
        current = by_generation.get(generation)
        if current is None or float(item.get("objective", 0.0)) < float(current.get("objective", 0.0)):
            by_generation[generation] = item

    rows = []
    for generation in sorted(by_generation.keys()):
        best = by_generation[generation]
        resources = best.get("resources", {})
        rows.append(
            {
                "generation": generation,
                "objective": float(best.get("objective", 0.0)),
                "perf_loss": float(best.get("perf_loss", 0.0)),
                "dsp": float(resources.get("dsp", 0.0)),
                "bram": float(resources.get("bram", 0.0)),
                "lat": float(resources.get("lat", 0.0)),
                "eng": float(resources.get("eng", 0.0)),
                "penalty_dsp": float(best.get("penalty_dsp", 0.0)),
                "penalty_bram": float(best.get("penalty_bram", 0.0)),
            }
        )

    if len(rows) == 0:
        logger.warning("No generation metadata found in NAS history; skip search log export.")
        return
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Exported NAS trajectory to {}", out_csv)


@logger.catch
def main(exp: BaseExp, args):
    if args.seed is not None:
        exp.seed = args.seed
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on CUDNN deterministic setting."
        )
    cudnn.benchmark = True
    
    output_dir = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logger to save to file
    logger.add(os.path.join(output_dir, "train_log.txt"))

    model = exp.get_model()
    
    # Load warm-start checkpoint if provided
    if args.ckpt is not None:
        maybe_load_ckpt(model, args.ckpt, device="cpu")
        
    def run_baseline():
        # Lightweight baseline stage placeholder.
        return {"status": 1.0, "note": 0.0}

    def run_qat(_model: torch.nn.Module):
        if args.skip_qat:
            return {"status": 0.0}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        exp.model = _model
        _model.to(device)
        _model.train()

        # Use an isolated optimizer for short QAT to avoid side effects from exp optimizer states.
        optimizer = _build_qat_optimizer(_model, lr=float(args.qat_lr))
        scaler = torch.amp.GradScaler('cuda', enabled=(device.startswith("cuda")))

        # If qat_epochs is 0, skip training loop and just evaluate
        if args.qat_epochs == 0:
            logger.info("QAT epochs is 0, running evaluation only.")
            _model.eval()
            evaluator = exp.get_evaluator(args.batch_size, is_distributed=False)
            with torch.no_grad():
                eval_result = evaluator.evaluate(_model, False, half=False)
            best_ap, best_ap50 = _parse_eval_ap(eval_result)
            return {
                "status": 1.0,
                "qat_epochs": 0.0,
                "avg_loss": 0.0,
                "best_ap": float(max(best_ap, 0.0)),
                "best_ap50": float(max(best_ap50, 0.0)),
                "non_finite_steps": 0.0,
                "qat_ckpt": args.ckpt,
            }

        train_loader = exp.get_data_loader(
            batch_size=args.batch_size,
            is_distributed=False,
            no_aug=True,
            cache_img=None,
        )
        prefetcher = DataPrefetcher(train_loader)
        max_iter = len(train_loader)
        if max_iter <= 0:
            raise RuntimeError("QAT train loader is empty.")

        if hasattr(exp, "get_lr_scheduler"):
            lr_scheduler = exp.get_lr_scheduler(float(args.qat_lr), max_iter)
        else:
            lr_scheduler = None

        evaluator = exp.get_evaluator(args.batch_size, is_distributed=False)
        best_ap = -1.0
        best_ap50 = -1.0
        best_state = copy.deepcopy(_model.state_dict())
        total_non_finite = 0
        rolling_loss = 0.0
        updates = 0
        data_type = torch.float16 if device.startswith("cuda") else torch.float32

        from tqdm import tqdm
        for epoch in range(max(1, int(args.qat_epochs))):
            # Check if we need to freeze BN at this epoch
            if args.freeze_bn_epoch >= 0 and epoch == args.freeze_bn_epoch:
                logger.info(f"Freezing Batch Normalization statistics at epoch {epoch}.")
                for m in _model.modules():
                    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                        m.eval()

            prefetcher = DataPrefetcher(train_loader)
            last_good_state = copy.deepcopy(_model.state_dict())
            
            pbar = tqdm(range(max_iter), desc=f"QAT Epoch {epoch+1}/{int(args.qat_epochs)}")
            for _ in pbar:
                inps, targets = prefetcher.next()
                inps = inps.to(device=device, dtype=data_type)
                targets = targets.to(device=device, dtype=data_type)
                targets.requires_grad = False
                inps, targets = exp.preprocess(inps, targets, exp.input_size)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device.startswith("cuda"))):
                    outputs = _model(inps, targets)
                    loss = outputs["total_loss"]

                if not torch.isfinite(loss):
                    total_non_finite += 1
                    _model.load_state_dict(last_good_state, strict=False)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.qat_grad_clip > 0:
                    clip_grad_norm_(_model.parameters(), max_norm=float(args.qat_grad_clip))
                scaler.step(optimizer)
                scaler.update()
                last_good_state = copy.deepcopy(_model.state_dict())

                if _is_spike_enabled(getattr(exp, "use_spike", False)):
                    from spikingjelly.activation_based import functional

                    functional.reset_net(_model)

                if lr_scheduler is not None:
                    lr_now = lr_scheduler.update_lr(epoch * max_iter + updates + 1)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_now

                rolling_loss += float(loss.detach().item())
                updates += 1
                
                # Update progress bar
                avg_loss_current = rolling_loss / max(1, updates)
                pbar.set_postfix({"loss": f"{avg_loss_current:.4f}"})

            should_eval = ((epoch + 1) % max(1, int(args.qat_eval_interval)) == 0) or (epoch + 1 == int(args.qat_epochs))
            if should_eval:
                _model.eval()
                with torch.no_grad():
                    eval_result = evaluator.evaluate(_model, False, half=False)
                epoch_ap, epoch_ap50 = _parse_eval_ap(eval_result)
                if epoch_ap >= best_ap:
                    best_ap = epoch_ap
                    best_ap50 = epoch_ap50
                    best_state = copy.deepcopy(_model.state_dict())
                _model.train()
                # Re-apply BN freeze if we are past the freeze epoch
                if args.freeze_bn_epoch >= 0 and epoch >= args.freeze_bn_epoch:
                    for m in _model.modules():
                        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                            m.eval()

        _model.load_state_dict(best_state, strict=False)
        avg_loss = rolling_loss / max(1, updates)
        ckpt_path = os.path.join(output_dir, "qat_best_ckpt.pth")
        torch.save({"model": best_state, "best_ap": best_ap, "best_ap50": best_ap50, "avg_loss": avg_loss}, ckpt_path)
        return {
            "status": 1.0,
            "qat_epochs": float(args.qat_epochs),
            "avg_loss": float(avg_loss),
            "best_ap": float(max(best_ap, 0.0)),
            "best_ap50": float(max(best_ap50, 0.0)),
            "non_finite_steps": float(total_non_finite),
            "qat_ckpt": ckpt_path,
        }

    # Apply HAPQ overrides if provided
    if args.hapq_force_bitwidth_w is not None or args.hapq_force_bitwidth_u is not None:
        if hasattr(exp, "get_hapq_config"):
            original_get_config = exp.get_hapq_config
            def get_overridden_config():
                cfg = original_get_config()
                if args.hapq_force_bitwidth_w is not None:
                    cfg["bit_choices_w"] = [args.hapq_force_bitwidth_w]
                if args.hapq_force_bitwidth_u is not None:
                    cfg["bit_choices_u"] = [args.hapq_force_bitwidth_u]
                return cfg
            exp.get_hapq_config = get_overridden_config
            logger.info(f"Overriding HAPQ config: W={args.hapq_force_bitwidth_w}, U={args.hapq_force_bitwidth_u}")
        else:
            logger.warning("Experiment has no get_hapq_config method, cannot override bitwidths.")

    pipeline = HAPQPipeline(
        exp=exp,
        run_baseline=None if args.skip_baseline else run_baseline,
        run_qat=run_qat,
    )

    result = pipeline.run(model=model, output_dir=output_dir)
    
    # Clean up hooks and manager if attached to model
    if hasattr(model, "_hapq_manager"):
        getattr(model, "_hapq_manager").clear_hooks()
        delattr(model, "_hapq_manager")
    if args.search_log_csv:
        export_search_log(result.history, args.search_log_csv)
    if args.output_json is not None and args.output_json != result.output_json:
        with open(result.output_json, "r", encoding="utf-8") as src, open(args.output_json, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        logger.info("HAPQ summary copied to {}", args.output_json)
    logger.info("HAPQ pipeline done. Summary: {}", result.output_json)


if __name__ == "__main__":
    configure_module()
    parser = make_parser()
    args = parser.parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.enable_hapq = True
    exp.merge(args.opts)
    check_exp_value(exp)
    if not args.experiment_name:
        args.experiment_name = f"{exp.exp_name}_hapq"
    main(exp, args)
