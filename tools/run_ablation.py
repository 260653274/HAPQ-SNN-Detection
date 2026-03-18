#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List

MODES = ["prune_only", "quant_w", "quant_wu", "full", "hapq"]

def main():
    parser = argparse.ArgumentParser("Run HAPQ ablation study (Pruning / Quant-W / Quant-WU / Full)")
    parser.add_argument("-f", "--exp_file", required=True, type=str, help="experiment description file")
    parser.add_argument("-n", "--name", default=None, type=str, help="model name")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="warm start checkpoint")
    parser.add_argument("--experiment-prefix", default="gen1_hapq_ablation", type=str, help="experiment name prefix")
    parser.add_argument("--modes", default="prune_only,quant_w,quant_wu,full", type=str, help="comma separated modes")
    parser.add_argument("--train-script", default="tools/train_hapq_event.py", type=str, help="train script path")
    parser.add_argument("--result-csv", required=True, type=str, help="output csv for ablation summary")
    parser.add_argument("--summary-dir", default="experiments/gen1_hapq/ablation", type=str, help="directory for run jsons")
    parser.add_argument("--extra-opts", nargs=argparse.REMAINDER, default=[], help="extra exp opts")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    os.makedirs(args.summary_dir, exist_ok=True)

    rows: List[Dict] = []

    for mode in modes:
        if mode not in MODES:
            print(f"Warning: Unknown ablation mode {mode}, skipping.")
            continue
        
        exp_name = f"{args.experiment_prefix}_{mode}"
        summary_path = os.path.join(args.summary_dir, f"{exp_name}.json")

        cmd = [
            sys.executable,
            args.train_script,
            "-f", args.exp_file,
            "-b", str(args.batch_size),
            "--experiment-name", exp_name,
            "--output-json", summary_path,
        ]
        if args.name:
            cmd += ["-n", args.name]
        if args.ckpt:
            cmd += ["-c", args.ckpt]
        
        if args.extra_opts:
            cmd += args.extra_opts

        # Set apply_mode via hapq_stage opt
        if mode == "hapq":
             # "hapq" mode typically means the "full" pipeline result from the main experiment
             cmd += ["hapq_stage", "full"]
        elif mode == "full":
             # "full" mode in ablation means SNN-FP (Baseline), so no HAPQ features
             cmd += ["hapq_stage", "baseline"]
        else:
             cmd += ["hapq_stage", mode]

        # Check if we should use existing checkpoint for eval-only
        # If user passes --qat-epochs 0 in extra-opts, we assume they want eval-only
        # But we need to point --ckpt to the previously trained checkpoint for this mode
        
        # Construct expected checkpoint path for this mode
        # Default output dir structure: YOLOX_outputs/{exp_name}/qat_best_ckpt.pth
        # We need to find where the previous run stored its checkpoint.
        # tools/train_hapq_event.py uses exp.output_dir / args.experiment_name
        # We can't easily know exp.output_dir here without parsing exp file, 
        # but we can guess standard YOLOX_outputs location.
        
        # However, to be safe and simple:
        # If the user wants eval-only, they should ensure the ckpt passed to this script 
        # is the one they want to evaluate. 
        # BUT, run_ablation iterates over modes. 
        # If we want to re-eval ALREADY TRAINED modes, we should look for their specific ckpts.
        
        mode_ckpt = os.path.join("YOLOX_outputs", exp_name, "qat_best_ckpt.pth")
        if mode == "hapq":
             # Special case for HAPQ main result
             mode_ckpt = "YOLOX_outputs/e_yolox_s_hapq_target_035/hapq_best_ckpt.pth"
        elif mode == "full":
             # "full" mode (Baseline) should always use the base checkpoint, 
             # not a potentially stale ablation checkpoint.
             mode_ckpt = args.ckpt

        if "--qat-epochs" in args.extra_opts:
            try:
                idx = args.extra_opts.index("--qat-epochs")
                if int(args.extra_opts[idx+1]) == 0:
                    if os.path.exists(mode_ckpt):
                        print(f"Eval-only mode detected. Using existing checkpoint: {mode_ckpt}")
                        # Override the generic ckpt with the mode-specific trained ckpt
                        # Find and replace -c/--ckpt in cmd if it exists, or append it
                        if "-c" in cmd:
                            c_idx = cmd.index("-c")
                            cmd[c_idx+1] = mode_ckpt
                        elif "--ckpt" in cmd:
                            c_idx = cmd.index("--ckpt")
                            cmd[c_idx+1] = mode_ckpt
                        else:
                            cmd += ["-c", mode_ckpt]
                    else:
                        print(f"Warning: Eval-only requested but checkpoint {mode_ckpt} not found. Using default/provided ckpt.")
            except (ValueError, IndexError):
                pass

        print(f"Running Ablation Mode: {mode} -> {exp_name}")
        subprocess.run(cmd, check=True)

        # Parse result
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            stage_metrics = data.get("stage_metrics", {})
            qat_metrics = stage_metrics.get("qat", {})
            pruning_metrics = stage_metrics.get("pruning_quantization", {})
            
            row = {
                "mode": mode,
                "experiment": exp_name,
                "mAP": qat_metrics.get("best_ap", 0.0),
                "mAP50": qat_metrics.get("best_ap50", 0.0),
                "Params": pruning_metrics.get("params", 0.0),
                "SynOps": pruning_metrics.get("synops", 0.0),
                "keep_ratio": pruning_metrics.get("keep_ratio", 1.0),
                "summary_json": summary_path,
            }
            rows.append(row)

    # Save summary CSV
    if rows:
        os.makedirs(os.path.dirname(args.result_csv) or ".", exist_ok=True)
        with open(args.result_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Ablation study completed. Summary saved to {args.result_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
