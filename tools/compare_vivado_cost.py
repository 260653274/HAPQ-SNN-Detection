#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import csv
import json
import math
import re
import os
from typing import Dict, List

from yolox.hapq.cost_model import HAPQBudget, HAPQCostModel, LayerCostSpec


def _to_float(raw: str) -> float:
    return float(raw.replace(",", "").strip())


def _parse_resource_line(report: str, keys: List[str]) -> float:
    for line in report.splitlines():
        # Check if ALL keys are present (case-insensitive)
        if not all(k.lower() in line.lower() for k in keys):
            continue
        # Extract numbers
        nums = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", line)
        if nums:
            return _to_float(nums[0])
    return math.nan


def parse_utilization_report(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return {
        "lut": _parse_resource_line(text, ["LUT", "Logic"]), # Match "LUT as Logic"
        "ff": _parse_resource_line(text, ["Register", "Flip", "Flop"]), # Match "Register as Flip Flop"
        "dsp": _parse_resource_line(text, ["DSPs"]), # Match "DSPs"
        "bram": _parse_resource_line(text, ["Block", "RAM", "Tile"]), # Match "Block RAM Tile"
    }


def parse_timing_report(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    wns = math.nan
    tns = math.nan

    # Look for "Design Timing Summary" table
    # It has a header line with WNS(ns), TNS(ns), etc.
    # And the data line follows (possibly with separators)
    
    for i, line in enumerate(lines):
        if "WNS(ns)" in line and "TNS(ns)" in line:
            # Found header. Look at subsequent lines for data.
            # Skip separator lines (starting with -)
            for j in range(i + 1, len(lines)):
                data_line = lines[j].strip()
                if not data_line or data_line.startswith("-"):
                    continue
                
                # This should be the data line
                parts = data_line.split()
                if len(parts) >= 2:
                    try:
                        wns = float(parts[0])
                        tns = float(parts[1])
                    except ValueError:
                        pass
                break
            break
            
    return {"wns": wns, "tns": tns}


def parse_power_report(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    power_data = {
        "total_power": math.nan,
        "dynamic_power": math.nan,
        "static_power": math.nan,
    }

    # Total On-Chip Power (W) | 1.234
    m = re.search(r"Total On-Chip Power\s+\(W\)\s*\|\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        power_data["total_power"] = float(m.group(1))
    
    # Dynamic (W) | 0.987
    m = re.search(r"Dynamic\s+\(W\)\s*\|\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        power_data["dynamic_power"] = float(m.group(1))

    # Device Static (W) | 0.247
    m = re.search(r"Device Static\s+\(W\)\s*\|\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        power_data["static_power"] = float(m.group(1))

    return power_data


def estimate_bandwidth(candidate: Dict, budget: HAPQBudget) -> float:
    # Bandwidth estimation based on state memory access:
    # BW = (Sum(State_Bits) * T_steps) / Latency_Target?
    # Or simpler: Total Bytes transferred per inference = Sum(Channels * H * W * b_u / 8) * T * 2 (Read+Write)
    # Here we just calculate Total State Bytes Access per Inference as a proxy
    
    total_state_bits = 0.0
    layers = candidate.get("layers", [])
    
    # Global T assumed from first layer or 4
    t_steps = 4
    if layers:
         t_steps = int(layers[0].get("timesteps", 4))
         
    for layer in layers:
        # Assuming spatial dims H*W are implicit in 'channels' or we rely on cost model specs
        # state_neurons = channels * H * W. The 'state_neurons' in cost model spec seems to be just channels*4?
        # Let's check _candidate_specs. 
        # _candidate_specs uses: state_neurons = channels * 4. This implies 4 is H*W? Unlikely.
        # It's likely a simplification.
        # Let's use the same logic as _candidate_specs for consistency, 
        # but the prompt asks for "Bytes_U equation".
        
        # In HAPQCostModel (implied), BW is likely modeled.
        # Let's use: Total State Bits * T_steps * 2 (Read+Write)
        
        channels = int(layer.get("channels", 64))
        b_u = int(layer.get("b_u", 12))
        
        # If we don't have H/W, we might be estimating per-pixel or per-block bandwidth
        # User prompt says: "add a placeholder/formula based on the paper's Bytes_U equation"
        
        # Let's assume a standard formula:
        # BW_per_sample = Sum(Neurons * b_u) * T * 2 (R+W) / 8 (bits->bytes)
        # But we need Neurons count. 
        # Using state_neurons from candidate specs if available, otherwise guess.
        
        neurons = float(layer.get("output_h", 1)) * float(layer.get("output_w", 1)) * channels
        if neurons == channels: # If H/W not in json, might be 1D or missing.
             # Fallback: assume some nominal size or just sum(channels)
             pass
        
        total_state_bits += neurons * b_u

    return (total_state_bits * t_steps * 2) / 8.0 # Bytes per inference


def _candidate_specs(candidate: Dict) -> List[LayerCostSpec]:
    specs: List[LayerCostSpec] = []
    for layer in candidate.get("layers", []):
        channels = int(layer.get("channels", 64))
        kernel = int(layer.get("kernel", 3))
        depth = int(layer.get("depth", 1))
        p_req = float(channels * kernel * kernel * depth)
        dense_synops = p_req * 32.0
        total_blocks = max(1, int(layer.get("total_blocks", 1)))
        active_blocks = max(1, int(layer.get("active_blocks", 1)))
        keep_ratio = min(1.0, max(0.0, active_blocks / total_blocks))
        specs.append(
            LayerCostSpec(
                name=layer.get("name", "layer"),
                p_req=p_req,
                dense_synops=dense_synops,
                activity=float(layer.get("activity", 0.2)),
                mask_keep_ratio=keep_ratio,
                b_w=int(layer.get("b_w", 8)),
                b_u=int(layer.get("b_u", 12)),
                state_neurons=channels * 4,
                timesteps=4,
            )
        )
    return specs


def _pct_err(estimate: float, measured: float) -> float:
    if math.isnan(estimate) or math.isnan(measured):
        return math.nan
    base = abs(estimate) if abs(estimate) > 1e-9 else 1.0
    return (measured - estimate) / base * 100.0


def main():
    parser = argparse.ArgumentParser("Compare Vivado utilization/timing with HAPQ cost model")
    parser.add_argument("--summary-json", required=True, type=str, help="hapq_summary.json path")
    parser.add_argument("--util-rpt", required=True, type=str, help="utilization_synth.rpt path")
    parser.add_argument("--timing-rpt", required=True, type=str, help="timing_synth.rpt path")
    parser.add_argument("--power-rpt", default=None, type=str, help="power.rpt path")
    parser.add_argument("--out-json", default=None, type=str, help="optional output json path")
    parser.add_argument("--out-csv", default=None, type=str, help="optional output csv path")
    parser.add_argument("--clock-period", default=5.0, type=float, help="target clock period in ns (default: 5.0)")
    parser.add_argument("--cycles-per-step", default=1000, type=int, help="estimated cycles per timestep (default: 1000)")
    parser.add_argument("--fail-on-violation", action="store_true", default=False, help="return non-zero on timing violation")
    args = parser.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    cfg = data.get("hapq_cfg", {})
    candidate = data.get("best_candidate", {})
    budget = HAPQBudget(
        tau_lat=float(cfg.get("tau_lat", 2.5e7)),
        tau_eng=float(cfg.get("tau_eng", 1500.0)),
        tau_dsp=float(cfg.get("tau_dsp", 1600)),
        tau_bram=float(cfg.get("tau_bram", 1200)),
        tau_bw=float(cfg.get("tau_bw", 5e8)),
        tau_lut=float(cfg.get("tau_lut", 2e5)),
    )
    model = HAPQCostModel(
        budget=budget,
        lambda_lat=float(cfg.get("lambda_lat", 1.0)),
        lambda_eng=float(cfg.get("lambda_eng", 1.0)),
        lambda_dsp=float(cfg.get("lambda_dsp", 1.0)),
        lambda_bram=float(cfg.get("lambda_bram", 1.0)),
        lambda_bw=float(cfg.get("lambda_bw", 1.0)),
        lambda_lut=float(cfg.get("lambda_lut", 1.0)),
    )
    estimate = model.estimate(_candidate_specs(candidate))
    vivado_util = parse_utilization_report(args.util_rpt)
    timing = parse_timing_report(args.timing_rpt)
    
    power = {"total_power": math.nan, "dynamic_power": math.nan, "static_power": math.nan}
    if args.power_rpt and os.path.exists(args.power_rpt):
        power = parse_power_report(args.power_rpt)
        
    est_bw_bytes = estimate_bandwidth(candidate, budget)
    
    timing_ok = (not math.isnan(timing["wns"])) and timing["wns"] >= 0.0

    # Calculate derived metrics
    # Frequency (MHz) = 1000 / (Period - WNS)
    # If WNS is positive, period is met, freq can be higher. If negative, freq is lower.
    # Actually, achievable period = Target - WNS.
    achieved_period = args.clock_period - timing["wns"] if not math.isnan(timing["wns"]) else args.clock_period
    freq_mhz = 1000.0 / max(0.001, achieved_period)

    # Latency (ms) = Total Cycles * Period / 1e6
    # Total Cycles = T * Cycles_Per_Step (simplified)
    # Get T from candidate or default to 4
    t_steps = 4
    if candidate.get("layers"):
        t_steps = int(candidate["layers"][0].get("timesteps", 4))
    
    total_cycles = t_steps * args.cycles_per_step
    latency_ms = total_cycles * achieved_period / 1e6
    
    # Throughput (FPS) = 1000 / Latency_ms
    fps = 1000.0 / max(1e-9, latency_ms)
    
    # Energy (mJ) = Power (W) * Latency (s) * 1000
    # Use total power if available
    power_val = power.get("total_power", math.nan)
    energy_mj = power_val * (latency_ms / 1000.0) * 1000.0 if not math.isnan(power_val) else math.nan

    # Bandwidth (MB/s) = Bytes / Latency (s) / 1e6
    bw_mbs = (est_bw_bytes / max(1e-9, latency_ms / 1000.0)) / 1e6

    result = {
        "estimate": {
            "dsp": float(estimate.dsp),
            "bram": float(estimate.bram),
            "lut": float(estimate.lut),
            "bw_bytes": est_bw_bytes,
        },
        "vivado": {
            **vivado_util,
            **power,
            "freq_mhz": freq_mhz,
            "latency_ms": latency_ms,
            "fps": fps,
            "energy_mj": energy_mj,
            "bw_mbs": bw_mbs,
        },
        "error_pct": {
            "dsp": _pct_err(float(estimate.dsp), vivado_util["dsp"]),
            "bram": _pct_err(float(estimate.bram), vivado_util["bram"]),
            "lut": _pct_err(float(estimate.lut), vivado_util["lut"]),
        },
        "timing": {
            "wns": timing["wns"],
            "tns": timing["tns"],
            "timing_ok": timing_ok,
            "timing_violation": not timing_ok,
        },
    }

    print(json.dumps(result, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    if args.out_csv:
        row = {
            "est_dsp": result["estimate"]["dsp"],
            "est_bram": result["estimate"]["bram"],
            "est_lut": result["estimate"]["lut"],
            "est_bw_bytes": result["estimate"]["bw_bytes"],
            "vivado_dsp": result["vivado"]["dsp"],
            "vivado_bram": result["vivado"]["bram"],
            "vivado_lut": result["vivado"]["lut"],
            "vivado_ff": result["vivado"].get("ff", math.nan),
            "vivado_freq_mhz": result["vivado"]["freq_mhz"],
            "vivado_latency_ms": result["vivado"]["latency_ms"],
            "vivado_fps": result["vivado"]["fps"],
            "vivado_energy_mj": result["vivado"]["energy_mj"],
            "vivado_bw_mbs": result["vivado"]["bw_mbs"],
            "vivado_power_total": result["vivado"]["total_power"],
            "vivado_power_dyn": result["vivado"]["dynamic_power"],
            "vivado_power_static": result["vivado"]["static_power"],
            "err_dsp_pct": result["error_pct"]["dsp"],
            "err_bram_pct": result["error_pct"]["bram"],
            "err_lut_pct": result["error_pct"]["lut"],
            "wns": result["timing"]["wns"],
            "tns": result["timing"]["tns"],
            "timing_ok": int(result["timing"]["timing_ok"]),
        }
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)

    if args.fail_on_violation and not timing_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
