#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import csv
import json
import os
from typing import Dict, List


def flatten_candidate_metrics(item: Dict) -> Dict:
    candidate = item.get("candidate", {})
    layers = candidate.get("layers", [])
    if len(layers) == 0:
        return {}
    avg_keep = sum(layer["active_blocks"] / max(1, layer["total_blocks"]) for layer in layers) / len(layers)
    avg_bw = sum(layer["b_w"] for layer in layers) / len(layers)
    avg_bu = sum(layer["b_u"] for layer in layers) / len(layers)
    return {
        "objective": item.get("objective", 0.0),
        "perf_loss": item.get("perf_loss", 0.0),
        "lat": item.get("resources", {}).get("lat", 0.0),
        "eng": item.get("resources", {}).get("eng", 0.0),
        "dsp": item.get("resources", {}).get("dsp", 0.0),
        "bram": item.get("resources", {}).get("bram", 0.0),
        "synops": item.get("resources", {}).get("synops", 0.0),
        "kappa": avg_keep,
        "bW": avg_bw,
        "bU": avg_bu,
    }


def write_csv(rows: List[Dict], path: str) -> None:
    if len(rows) == 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser("Export HAPQ metrics for matplotlib")
    parser.add_argument("--summary-json", required=True, type=str, help="hapq_summary.json path")
    parser.add_argument("--out-csv", required=True, type=str, help="output csv path")
    parser.add_argument("--out-json", default=None, type=str, help="optional output json path")
    parser.add_argument("--topk", default=20, type=int, help="export top-k candidates")
    args = parser.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("history", [])
    rows = []
    for item in history[: args.topk]:
        row = flatten_candidate_metrics(item)
        if row:
            rows.append(row)
    write_csv(rows, args.out_csv)

    if args.out_json is not None:
        payload = {
            "rows": rows,
            "stage_metrics": data.get("stage_metrics", {}),
            "best_candidate": data.get("best_candidate", {}),
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
