#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Dict, List

TIERS = {
    "Tiny": {
        "tau_dsp": 300.0,
        "tau_bram": 250.0,
        "tau_lat": 4.0e7,
        "tau_eng": 800.0,
    },
    "Small": {
        "tau_dsp": 800.0,
        "tau_bram": 600.0,
        "tau_lat": 2.5e7,
        "tau_eng": 1500.0,
    },
    "Medium": {
        "tau_dsp": 1600.0,
        "tau_bram": 1200.0,
        "tau_lat": 1.5e7,
        "tau_eng": 3000.0,
    },
    "Large": {
        "tau_dsp": 2500.0,
        "tau_bram": 1800.0,
        "tau_lat": 1.0e7,
        "tau_eng": 5000.0,
    },
}

def main():
    parser = argparse.ArgumentParser("Sweep HAPQ hardware tiers (Tiny/Small/Medium/Large)")
    parser.add_argument("-f", "--exp_file", required=True, type=str, help="experiment description file")
    parser.add_argument("-n", "--name", default=None, type=str, help="model name")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="warm start checkpoint")
    parser.add_argument("--experiment-prefix", default="gen1_hapq_tier", type=str, help="experiment name prefix")
    parser.add_argument("--tiers", default="Tiny,Small,Medium,Large", type=str, help="comma separated tiers")
    parser.add_argument("--train-script", default="tools/train_hapq_event.py", type=str, help="train script path")
    parser.add_argument("--result-csv", required=True, type=str, help="output csv for tier summary")
    parser.add_argument("--summary-dir", default="experiments/gen1_hapq/tiers", type=str, help="directory for run jsons")
    parser.add_argument("--extra-opts", nargs=argparse.REMAINDER, default=[], help="extra exp opts")
    args = parser.parse_args()

    tier_names = [t.strip() for t in args.tiers.split(",") if t.strip()]
    os.makedirs(args.summary_dir, exist_ok=True)

    rows: List[Dict] = []

    for tier_name in tier_names:
        if tier_name not in TIERS:
            print(f"Warning: Unknown tier {tier_name}, skipping.")
            continue
        
        budget = TIERS[tier_name]
        exp_name = f"{args.experiment_prefix}_{tier_name.lower()}"
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
        
        # Add budget constraints
        cmd += [
            "hapq_tau_dsp", str(budget["tau_dsp"]),
            "hapq_tau_bram", str(budget["tau_bram"]),
            "hapq_tau_lat", str(budget["tau_lat"]),
            "hapq_tau_eng", str(budget["tau_eng"]),
        ]
        
        if args.extra_opts:
            cmd += args.extra_opts

        print(f"Running Tier: {tier_name} -> {exp_name}")
        subprocess.run(cmd, check=True)

        # Parse result
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            best_candidate = data.get("best_candidate", {})
            stage_metrics = data.get("stage_metrics", {})
            qat_metrics = stage_metrics.get("qat", {})
            
            # Extract metrics
            row = {
                "tier": tier_name,
                "experiment": exp_name,
                "mAP": qat_metrics.get("best_ap", 0.0),
                "lat_est": float(best_candidate.get("metadata", {}).get("lat", 0.0)),
                "eng_est": float(best_candidate.get("metadata", {}).get("eng", 0.0)),
                "dsp_est": float(best_candidate.get("metadata", {}).get("dsp", 0.0)),
                "bram_est": float(best_candidate.get("metadata", {}).get("bram", 0.0)),
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
        print(f"Tier sweep completed. Summary saved to {args.result_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
