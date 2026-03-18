#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import csv
import json
import os
import subprocess
import sys
import shutil

def run_command(cmd, dry_run=False):
    print(f"Running: {' '.join(cmd)}")
    if not dry_run:
        subprocess.check_call(cmd)

def main():
    parser = argparse.ArgumentParser("Run FPGA implementation flow for HAPQ experiments")
    parser.add_argument("--input-csv", required=True, type=str, help="ablation_summary.csv or similar")
    parser.add_argument("--output-csv", default="fpga_results.csv", type=str, help="final aggregated results")
    parser.add_argument("--vivado-mode", default="batch", choices=["batch", "gui"], help="Vivado mode")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--tcl-script", default="fpga/scripts/synth_hapq.tcl", type=str, help="Tcl script path")
    parser.add_argument("--clock-period", default=5.0, type=float, help="target clock period in ns")
    parser.add_argument("--cycles-per-step", default=1000, type=int, help="estimated cycles per timestep")
    parser.add_argument("--vivado-path", default=None, type=str, help="Path to Vivado executable (e.g., /opt/Xilinx/Vivado/2025.2/bin/vivado)")
    args = parser.parse_args()

    # Verify input exists
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV {args.input_csv} not found.")
        sys.exit(1)

    results = []
    
    with open(args.input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Vivado project paths
    proj_dir = os.path.abspath("fpga/vivado/eas_snn_hapq")
    util_rpt = os.path.join(proj_dir, "utilization_impl.rpt")
    timing_rpt = os.path.join(proj_dir, "timing_impl.rpt")
    power_rpt = os.path.join(proj_dir, "power.rpt")
    
    # If impl reports don't exist (e.g. only synth run), fallback to synth? 
    # The tcl script I wrote does both and reports both. I'll prefer impl.

    for row in rows:
        exp_name = row.get("experiment", "unknown")
        summary_json = row.get("summary_json")
        
        if not summary_json or not os.path.exists(summary_json):
            print(f"Skipping {exp_name}: summary_json not found ({summary_json})")
            continue

        print(f"\n=== Processing Experiment: {exp_name} ===")
        
        # 1. Export Params
        cmd_export = [
            sys.executable,
            "tools/export_hapq_verilog_params.py",
            "--summary-json", summary_json,
            "--out-vh", "fpga/rtl/params.vh"
        ]
        run_command(cmd_export, args.dry_run)

        # 2. Run Vivado
        # Check if vivado is available
        vivado_bin = None
        settings_sh = None
        
        # Priority 1: User provided path
        if args.vivado_path:
            vivado_bin = args.vivado_path
        
        # Priority 2: PATH
        if not vivado_bin and shutil.which("vivado"):
            vivado_bin = shutil.which("vivado")
            
        # Priority 3: Common locations
        if not vivado_bin:
            common_paths = [
                "/opt/Xilinx/2025.2/Vivado/bin/vivado",
                "/tools/Xilinx/2025.2/Vivado/bin/vivado",
                "/opt/Xilinx/Vivado/2025.2/bin/vivado",
                "/tools/Xilinx/Vivado/2025.2/bin/vivado",
                "/opt/Xilinx/Vivado/2024.1/bin/vivado",
                "/tools/Xilinx/Vivado/2024.1/bin/vivado",
            ]
            for path in common_paths:
                if os.path.exists(path):
                    vivado_bin = path
                    print(f"Found Vivado at: {vivado_bin}")
                    break
        
        if not vivado_bin:
            print("Vivado not found in PATH or common locations. Skipping execution (Simulated run).")
            if not args.dry_run:
                 print("Warning: Cannot run Vivado. Expecting reports to be present or this will fail.")
        else:
            # Try to find settings64.sh relative to bin
            # Expected structure: .../Vivado/2025.2/bin/vivado -> .../Vivado/2025.2/settings64.sh
            bin_dir = os.path.dirname(vivado_bin)
            install_dir = os.path.dirname(bin_dir) # .../Vivado/2025.2
            potential_settings = os.path.join(install_dir, "settings64.sh")
            
            cmd_vivado = []
            if os.path.exists(potential_settings):
                print(f"Sourcing settings from: {potential_settings}")
                # Use bash to source settings then run vivado
                # We need to construct the full command line for vivado inside the bash string
                vivado_args = f"-mode {args.vivado_mode} -source {args.tcl_script} -notrace"
                cmd_vivado = [
                    "bash", "-c",
                    f"source {potential_settings} && {vivado_bin} {vivado_args}"
                ]
            else:
                cmd_vivado = [
                    vivado_bin,
                    "-mode", args.vivado_mode,
                    "-source", args.tcl_script,
                    "-notrace"
                ]
            
            run_command(cmd_vivado, args.dry_run)

        # 3. Parse Results
        # specific output json for this run
        out_json_path = summary_json.replace(".json", "_fpga_metrics.json")
        
        cmd_parse = [
            sys.executable,
            "tools/compare_vivado_cost.py",
            "--summary-json", summary_json,
            "--util-rpt", util_rpt,
            "--timing-rpt", timing_rpt,
            "--power-rpt", power_rpt,
            "--clock-period", str(args.clock_period),
            "--cycles-per-step", str(args.cycles_per_step),
            "--out-json", out_json_path
        ]
        
        if args.dry_run:
            print(f"Running: {' '.join(cmd_parse)}")
            # Mock result
            parsed_data = {}
        else:
            # We need to handle the case where reports might not exist if Vivado failed or wasn't run
            if os.path.exists(util_rpt):
                run_command(cmd_parse, False)
                if os.path.exists(out_json_path):
                    with open(out_json_path, "r", encoding="utf-8") as f:
                        parsed_data = json.load(f)
                else:
                    parsed_data = {}
            else:
                print(f"Warning: Reports not found for {exp_name}")
                parsed_data = {}

        # 4. Aggregate
        # Merge original row with new metrics
        new_row = dict(row)
        
        vivado_res = parsed_data.get("vivado", {})
        timing_res = parsed_data.get("timing", {})
        est_res = parsed_data.get("estimate", {})
        
        new_row["FPGA_LUT"] = vivado_res.get("lut", "")
        new_row["FPGA_FF"] = vivado_res.get("ff", "")
        new_row["FPGA_DSP"] = vivado_res.get("dsp", "")
        new_row["FPGA_BRAM"] = vivado_res.get("bram", "")
        new_row["FPGA_Power_W"] = vivado_res.get("total_power", "")
        new_row["FPGA_Power_Dyn_W"] = vivado_res.get("dynamic_power", "")
        new_row["FPGA_WNS"] = timing_res.get("wns", "")
        new_row["FPGA_Freq_MHz"] = vivado_res.get("freq_mhz", "")
        new_row["FPGA_Latency_ms"] = vivado_res.get("latency_ms", "")
        new_row["FPGA_FPS"] = vivado_res.get("fps", "")
        new_row["FPGA_Energy_mJ"] = vivado_res.get("energy_mj", "")
        new_row["FPGA_BW_MBs"] = vivado_res.get("bw_mbs", "")
        
        new_row["Est_BW_Bytes"] = est_res.get("bw_bytes", "")
        
        results.append(new_row)

    # Save aggregated CSV
    if results:
        keys = list(results[0].keys())
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Aggregated results saved to {args.output_csv}")
    else:
        print("No results processed.")

if __name__ == "__main__":
    main()
