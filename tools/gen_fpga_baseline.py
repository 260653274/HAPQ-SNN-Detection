#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import json
import os
import copy

def main():
    parser = argparse.ArgumentParser("Generate FPGA Baseline Configuration (Full Precision, No Pruning)")
    parser.add_argument("--template-json", required=True, type=str, help="Existing hapq_summary.json to use as architecture template")
    parser.add_argument("--out-json", default="experiments/baseline/baseline_summary.json", type=str, help="Output baseline json")
    parser.add_argument("--bitwidth", default=16, type=int, help="Baseline bitwidth (default 16)")
    args = parser.parse_args()

    if not os.path.exists(args.template_json):
        print(f"Error: Template {args.template_json} not found.")
        return

    with open(args.template_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # We modify the best_candidate to be fully dense and full precision
    candidate = data.get("best_candidate", {})
    if not candidate:
        print("Error: No best_candidate in template.")
        return

    baseline_candidate = copy.deepcopy(candidate)
    
    for layer in baseline_candidate.get("layers", []):
        # 1. Full Precision
        layer["b_w"] = args.bitwidth
        layer["b_u"] = args.bitwidth
        
        # 2. No Pruning
        total = layer.get("total_blocks", 1)
        layer["active_blocks"] = total
        layer["activity"] = 1.0 # Assumption: Baseline might not have sparsity advantages? 
                                # But activity comes from data. 
                                # For HW cost model, if we use dense ops, activity doesn't reduce static allocation.
                                # But dynamic power might be affected. 
                                # However, 'No Pruning' means we allocate for full blocks.
        
    # Update config to reflect baseline intent if needed
    data["hapq_cfg"]["experiment_name"] = "FPGA_Baseline_FP16"
    data["best_candidate"] = baseline_candidate
    
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Baseline configuration generated at: {args.out_json}")
    print(f"Settings: Bitwidth={args.bitwidth}, Pruning=None (100% Active)")

if __name__ == "__main__":
    main()
