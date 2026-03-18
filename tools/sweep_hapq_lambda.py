#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple


def parse_list(raw: str) -> List[float]:
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if len(items) == 0:
        raise ValueError("Lambda list cannot be empty.")
    return [float(x) for x in items]


def summarize_budget(best_item: Dict, cfg: Dict) -> Tuple[bool, Dict[str, float]]:
    resources = best_item.get("resources", {})
    checks = {
        "dsp": float(resources.get("dsp", 0.0)) <= float(cfg.get("tau_dsp", float("inf"))),
        "bram": float(resources.get("bram", 0.0)) <= float(cfg.get("tau_bram", float("inf"))),
        "lat": float(resources.get("lat", 0.0)) <= float(cfg.get("tau_lat", float("inf"))),
        "eng": float(resources.get("eng", 0.0)) <= float(cfg.get("tau_eng", float("inf"))),
        "bw": float(resources.get("bw", 0.0)) <= float(cfg.get("tau_bw", float("inf"))),
        "lut": float(resources.get("lut", 0.0)) <= float(cfg.get("tau_lut", float("inf"))),
    }
    return all(checks.values()), {k: float(v) for k, v in resources.items() if isinstance(v, (float, int))}


def main():
    parser = argparse.ArgumentParser("Sweep HAPQ lambda_dsp/lambda_bram on Gen1")
    parser.add_argument("-f", "--exp_file", required=True, type=str, help="experiment description file")
    parser.add_argument("-n", "--name", default=None, type=str, help="model name")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="warm start checkpoint")
    parser.add_argument("--experiment-prefix", default="gen1_hapq_sweep", type=str, help="experiment name prefix")
    parser.add_argument("--lambda-dsp", default="1,2,4,8", type=str, help="comma separated lambda_dsp values")
    parser.add_argument("--lambda-bram", default="1,2,4,8", type=str, help="comma separated lambda_bram values")
    parser.add_argument("--train-script", default="tools/train_hapq_event.py", type=str, help="train script path")
    parser.add_argument("--result-csv", required=True, type=str, help="output csv for sweep summary")
    parser.add_argument("--summary-dir", default="experiments/gen1_hapq/sweep", type=str, help="directory for run jsons")
    parser.add_argument("--extra-opts", nargs=argparse.REMAINDER, default=[], help="extra exp opts")
    args = parser.parse_args()

    lambda_dsp_list = parse_list(args.lambda_dsp)
    lambda_bram_list = parse_list(args.lambda_bram)
    os.makedirs(args.summary_dir, exist_ok=True)

    rows: List[Dict] = []
    best_feasible = None
    best_any = None
    for lambda_dsp, lambda_bram in itertools.product(lambda_dsp_list, lambda_bram_list):
        tag = f"ldsp{lambda_dsp:g}_lbram{lambda_bram:g}"
        exp_name = f"{args.experiment_prefix}_{tag}"
        summary_path = os.path.join(args.summary_dir, f"{exp_name}.json")

        cmd = [
            sys.executable,
            args.train_script,
            "-f",
            args.exp_file,
            "-b",
            str(args.batch_size),
            "--experiment-name",
            exp_name,
            "--skip-baseline",
            "--skip-qat",
            "--output-json",
            summary_path,
        ]
        if args.name:
            cmd += ["-n", args.name]
        if args.ckpt:
            cmd += ["-c", args.ckpt]
        cmd += ["hapq_lambda_dsp", str(lambda_dsp), "hapq_lambda_bram", str(lambda_bram)]
        if args.extra_opts:
            cmd += args.extra_opts

        subprocess.run(cmd, check=True)
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        history = data.get("history", [])
        best_item = history[0] if history else {}
        cfg = data.get("hapq_cfg", {})
        feasible, resources = summarize_budget(best_item, cfg)
        objective = float(best_item.get("objective", 1e9))
        row = {
            "experiment": exp_name,
            "lambda_dsp": lambda_dsp,
            "lambda_bram": lambda_bram,
            "objective": objective,
            "perf_loss": float(best_item.get("perf_loss", 0.0)),
            "feasible": int(feasible),
            "dsp": float(resources.get("dsp", 0.0)),
            "bram": float(resources.get("bram", 0.0)),
            "lat": float(resources.get("lat", 0.0)),
            "eng": float(resources.get("eng", 0.0)),
            "bw": float(resources.get("bw", 0.0)),
            "lut": float(resources.get("lut", 0.0)),
            "summary_json": summary_path,
        }
        rows.append(row)
        if best_any is None or objective < best_any["objective"]:
            best_any = row
        if feasible and (best_feasible is None or objective < best_feasible["objective"]):
            best_feasible = row

    rows.sort(key=lambda x: (1 - int(x["feasible"]), float(x["objective"])))
    os.makedirs(os.path.dirname(args.result_csv) or ".", exist_ok=True)
    with open(args.result_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["experiment"])
        writer.writeheader()
        writer.writerows(rows)

    if best_feasible is not None:
        print("Best feasible:", best_feasible["experiment"], "objective=", best_feasible["objective"])
    elif best_any is not None:
        print("No feasible candidate found, best overall:", best_any["experiment"], "objective=", best_any["objective"])


if __name__ == "__main__":
    main()
