#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List

def main():
    parser = argparse.ArgumentParser("Run HAPQ Sensitivity Sweep (Bitwidth b_U)")
    parser.add_argument("-f", "--exp_file", required=True, type=str, help="experiment description file")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("-c", "--ckpt", default="ckpt/best_ckpt.pth", type=str)
    parser.add_argument("--experiment-prefix", default="gen1_sweep_bu", type=str)
    parser.add_argument("--bits", default="16,8,6,4", type=str, help="comma separated bitwidths for membrane potential")
    parser.add_argument("--train-script", default="tools/train_hapq_event.py", type=str)
    parser.add_argument("--result-csv", required=True, type=str)
    parser.add_argument("--summary-dir", default="experiments/gen1_hapq/sweep_bu", type=str)
    parser.add_argument("--extra-opts", nargs=argparse.REMAINDER, default=[], help="extra exp opts")
    args = parser.parse_args()

    bits = [int(b.strip()) for b in args.bits.split(",") if b.strip()]
    os.makedirs(args.summary_dir, exist_ok=True)
    rows: List[Dict] = []

    for b_u in bits:
        exp_name = f"{args.experiment_prefix}_u{b_u}"
        summary_path = os.path.join(args.summary_dir, f"{exp_name}.json")
        
        # Construct command overrides for fixed quantization
        cmd = [
            sys.executable,
            args.train_script,
            "-f", args.exp_file,
            "-b", str(args.batch_size),
            "--experiment-name", exp_name,
            "--output-json", summary_path,
            "-c", args.ckpt,
            "--hapq-force-bitwidth-u", str(b_u),
            "--hapq-force-bitwidth-w", "8", # Fixed weight bitwidth for isolation
        ]
        
        if args.extra_opts:
            cmd += args.extra_opts

        # Pass 'hapq_stage' as an opt to be parsed by exp.merge()
        # This tells HAPQPipeline to use 'quant_wu' mode logic
        cmd += ["hapq_stage", "quant_wu"]

        print(f"Running Sweep: b_U={b_u} -> {exp_name}")
        subprocess.run(cmd, check=True)

        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            stage_metrics = data.get("stage_metrics", {})
            qat_metrics = stage_metrics.get("qat", {})
            pruning_metrics = stage_metrics.get("pruning_quantization", {})

            rows.append({
                "b_U": b_u,
                "mAP": qat_metrics.get("best_ap", 0.0),
                "mAP50": qat_metrics.get("best_ap50", 0.0),
                "Params": pruning_metrics.get("params", 0.0),
                "experiment": exp_name
            })

    if rows:
        os.makedirs(os.path.dirname(args.result_csv) or ".", exist_ok=True)
        with open(args.result_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Sweep completed. Saved to {args.result_csv}")

if __name__ == "__main__":
    main()
