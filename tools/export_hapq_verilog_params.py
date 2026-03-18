#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import json
import os
from statistics import mean
from typing import Dict, List


def _int_mean(values: List[int], default_value: int) -> int:
    if len(values) == 0:
        return int(default_value)
    return int(round(mean(values)))


def derive_global_params(candidate: Dict, default_t_steps: int = 4) -> Dict[str, int]:
    layers = candidate.get("layers", [])
    if len(layers) == 0:
        raise ValueError("Candidate has no layers.")

    b_w = [int(layer.get("b_w", 8)) for layer in layers]
    b_u = [int(layer.get("b_u", 12)) for layer in layers]
    active_blocks = [int(layer.get("active_blocks", 1)) for layer in layers]
    total_blocks = [int(layer.get("total_blocks", 1)) for layer in layers]
    leak_shift = [int(layer.get("leak_shift_n", 3)) for layer in layers]

    # Use aggregate-safe defaults for global RTL macros.
    data_width = max(1, _int_mean(b_w, 8))
    u_width = max(1, _int_mean(b_u, 12))
    # Use active_blocks to dimension hardware to reflect pruning benefits
    blocks = max(1, max(active_blocks))
    shift_n = max(0, _int_mean(leak_shift, 3))
    t_steps = max(1, int(default_t_steps))
    return {
        "HAPQ_DATA_WIDTH": data_width,
        "HAPQ_U_WIDTH": u_width,
        "HAPQ_BLOCKS": blocks,
        "HAPQ_T_STEPS": t_steps,
        "HAPQ_SHIFT_N": shift_n,
    }


def render_vh(defines: Dict[str, int]) -> str:
    lines = ["`ifndef HAPQ_PARAMS_VH", "`define HAPQ_PARAMS_VH", ""]
    for key in ["HAPQ_DATA_WIDTH", "HAPQ_U_WIDTH", "HAPQ_BLOCKS", "HAPQ_T_STEPS", "HAPQ_SHIFT_N"]:
        lines.append(f"`define {key} {int(defines[key])}")
    lines += ["", "`endif", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser("Export best_candidate to Verilog params.vh")
    parser.add_argument("--summary-json", required=True, type=str, help="hapq_summary.json path")
    parser.add_argument("--out-vh", default="fpga/rtl/params.vh", type=str, help="output params.vh path")
    parser.add_argument("--t-steps", default=4, type=int, help="global T_STEPS for RTL")
    args = parser.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    candidate = data.get("best_candidate", data)
    defines = derive_global_params(candidate, default_t_steps=args.t_steps)
    payload = render_vh(defines)

    os.makedirs(os.path.dirname(args.out_vh) or ".", exist_ok=True)
    with open(args.out_vh, "w", encoding="utf-8") as f:
        f.write(payload)

    print("Exported params:", defines)
    print("Output:", args.out_vh)


if __name__ == "__main__":
    main()
