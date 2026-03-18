#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def parse_series_item(item: str):
    if "=" not in item:
        raise ValueError(f"Invalid series spec: {item}. Expected label=csv_path")
    label, path = item.split("=", 1)
    return label.strip(), path.strip()


def parse_stage_map(items: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid stage mAP spec: {item}. Expected label:map")
        label, raw = item.split(":", 1)
        out[label.strip()] = float(raw.strip())
    return out


def read_points(csv_path: str, label: str, stage_map: Dict[str, float], topk: int) -> List[Dict]:
    points: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if topk > 0 and idx >= topk:
                break
            objective = float(row.get("objective", 0.0))
            map_value = row.get("map", row.get("mAP", ""))
            if map_value == "":
                map_value = stage_map.get(label)
            else:
                map_value = float(map_value)
            if map_value is None:
                continue
            efficiency = 1.0 / max(1e-9, objective)
            points.append(
                {
                    "label": label,
                    "efficiency": efficiency,
                    "map": float(map_value),
                    "objective": objective,
                }
            )
    return points


def pareto_frontier(points: List[Dict]) -> List[Dict]:
    # Maximize both map and efficiency.
    sorted_points = sorted(points, key=lambda p: p["efficiency"], reverse=True)
    frontier = []
    best_map = -1e9
    for p in sorted_points:
        if p["map"] >= best_map:
            frontier.append(p)
            best_map = p["map"]
    frontier.sort(key=lambda p: p["efficiency"])
    return frontier


def main():
    parser = argparse.ArgumentParser("Plot mAP vs Efficiency trade-off for HAPQ")
    parser.add_argument(
        "--series",
        nargs="+",
        required=True,
        help="list of label=csv_path, e.g. SNN-HAPQ=out/hapq.csv SNN-QW=out/qw.csv",
    )
    parser.add_argument(
        "--stage-map",
        nargs="*",
        default=[],
        help="optional per-stage fixed mAP values: label:map, e.g. SNN-HAPQ:0.41",
    )
    parser.add_argument("--topk", default=20, type=int, help="max points per series from csv")
    parser.add_argument("--out-png", required=True, type=str, help="output png path")
    parser.add_argument("--out-pdf", default=None, type=str, help="optional output pdf path")
    parser.add_argument("--title", default="mAP vs Efficiency", type=str, help="plot title")
    args = parser.parse_args()

    stage_map = parse_stage_map(args.stage_map)
    all_points: List[Dict] = []
    for spec in args.series:
        label, path = parse_series_item(spec)
        all_points.extend(read_points(path, label, stage_map, args.topk))
    if len(all_points) == 0:
        raise RuntimeError("No valid points to plot. Ensure each series provides objective and mAP.")

    plt.figure(figsize=(8, 5))
    labels = sorted(list({p["label"] for p in all_points}))
    for label in labels:
        xs = [p["efficiency"] for p in all_points if p["label"] == label]
        ys = [p["map"] for p in all_points if p["label"] == label]
        plt.scatter(xs, ys, label=label, alpha=0.85, s=28)

    frontier = pareto_frontier(all_points)
    plt.plot([p["efficiency"] for p in frontier], [p["map"] for p in frontier], linewidth=1.5, label="Pareto")
    plt.xlabel("Efficiency (1 / objective)")
    plt.ylabel("mAP")
    plt.title(args.title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plt.savefig(args.out_png, dpi=180)
    if args.out_pdf:
        os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)
        plt.savefig(args.out_pdf)
    print("Saved:", args.out_png)
    if args.out_pdf:
        print("Saved:", args.out_pdf)


if __name__ == "__main__":
    main()
