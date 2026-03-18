# Gen1 HAPQ Closed-Loop Experiment

> **Hardware-Aware Pruning & Quantization (HAPQ)** pipeline for event-based object detection on the Gen1 dataset — covering training, NAS-driven mixed-precision search, QAT, ablation study, FPGA synthesis, and trade-off analysis.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Pipeline Overview](#pipeline-overview)
- [Step 1 — Training, Search & QAT](#step-1--training-search--qat)
- [Step 2 — Ablation Study](#step-2--ablation-study)
- [Step 3 — FPGA Hardware Evaluation](#step-3--fpga-hardware-evaluation)
- [Step 4 — Trade-off Visualization](#step-4--trade-off-visualization)
- [Step 5 — Compute-Tier Sweep](#step-5--compute-tier-sweep)
- [Step 6 — Membrane Potential Sensitivity Sweep](#step-6--membrane-potential-sensitivity-sweep)
- [Appendix — Manual FPGA Debug Flow](#appendix--manual-fpga-debug-flow)

---

## Prerequisites

| Requirement | Details |
|---|---|
| Conda environment | `eas-snn` |
| FPGA device | `xczu7ev-ffvc1156-2-i` (adjust as needed) |
| Board connection | JTAG + UART |
| Vivado | Must be available on `PATH` for FPGA steps |

---

## Environment Setup

Run the following at the start of every new terminal session:

```bash
cd /home/xlang/EAS-SNN
source ~/.bashrc
conda activate eas-snn

# Optional: verify CUDA availability
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Pipeline Overview

```
Training + Search + QAT
        │
        ▼
  Ablation Study  ──────────────────────────┐
        │                                   │
        ▼                                   ▼
FPGA Baseline Gen          FPGA Synthesis & Implementation
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
              Trade-off Visualization
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
  Compute-Tier Sweep     Sensitivity Sweep (b_U)
```

---

## Step 1 — Training, Search & QAT

Runs the full HAPQ pipeline: structured pruning search, mixed-precision NAS, and quantization-aware training (QAT). Results are saved to a new experiment folder without overwriting prior runs.

```bash
python tools/train_hapq_event.py \
    -f exps/default/e_yolox_s_hapq.py \
    -c ckpt/best_ckpt.pth \
    -b 16 \
    -expn e_yolox_s_hapq_target_035 \
    --skip-baseline \
    --freeze-bn-epoch 30 \
    --qat-lr 2e-5 \
    --qat-grad-clip 1.0 \
    --qat-epochs 50 \
    --output-json experiments/gen1_hapq/gen1_hapq_summary_target_035.json
```

**Outputs:**

| Path | Description |
|---|---|
| `experiments/gen1_hapq/gen1_hapq_summary_target_035.json` | Best HAPQ configuration |
| `YOLOX_outputs/e_yolox_s_hapq_target_035/hapq_best_ckpt.pth` | Best checkpoint |

---

## Step 2 — Ablation Study

Generates ablation variants (Prune-only, Quant-W, Quant-W+U, Full) and records key metrics (mAP50:95, mAP50, Params, SynOps). Corresponds to **Section A2–A3** of `LaTeX/exp.tex`.

**Ablation dimensions:**

| Dimension | Description |
|---|---|
| Is Structure Necessary? | Validates structured pruning over unstructured alternatives |
| Is State Quantization Safe? | Compares Quant-W (FP16 state) vs. Quant-W+U (INT8 state) |
| Is NAS Effective? | Validates mixed-precision search over fixed-precision strategies |

### Full Training Run

```bash
python tools/run_ablation.py \
  -f exps/default/e_yolox_s_hapq.py \
  --ckpt YOLOX_outputs/e_yolox_s_hapq_target_035/hapq_best_ckpt.pth \
  --experiment-prefix gen1_ablation \
  --modes prune_only,quant_w,quant_wu,full \
  --result-csv experiments/gen1_hapq/ablation/ablation_summary.csv \
  --summary-dir experiments/gen1_hapq/ablation \
  --extra-opts --qat-epochs 50 --qat-lr 2e-5 --qat-grad-clip 1.0
```

### Eval-Only Re-run (no retraining)

Requires checkpoints from a prior run. Adds `hapq` mode to include the main experiment result.

```bash
python tools/run_ablation.py \
  -f exps/default/e_yolox_s_hapq.py \
  --ckpt ckpt/best_ckpt.pth \
  --experiment-prefix gen1_ablation \
  --modes prune_only,quant_w,quant_wu,full,hapq \
  --result-csv experiments/gen1_hapq/ablation/ablation_summary.csv \
  --summary-dir experiments/gen1_hapq/ablation \
  --extra-opts --qat-epochs 0
```

**Outputs:**

| Path | Description |
|---|---|
| `experiments/gen1_hapq/ablation/ablation_summary.csv` | Summary table (mAP, Params, SynOps, KeepRatio, …) |
| `experiments/gen1_hapq/ablation/*.json` | Per-variant detailed configs and metrics |

---

## Step 3 — FPGA Hardware Evaluation

Automates the full FPGA flow: Verilog parameter export → Vivado synthesis/implementation → resource/power/bandwidth parsing → results aggregation.

### 3.1 Generate Baseline Configuration (SNN-FP16, No Pruning)

```bash
python tools/gen_fpga_baseline.py \
  --template-json experiments/gen1_hapq/gen1_hapq_summary_04.json \
  --out-json experiments/gen1_hapq/ablation/baseline_summary.json \
  --bitwidth 16
```

### 3.2 (Optional) Add Baseline to Ablation List

The baseline can be added to `ablation_summary.csv` for a unified flow, or run separately. The batch script below will skip any missing entries automatically.

### 3.3 Batch FPGA Synthesis & Implementation

> **Note:** Vivado must be available on `PATH`. Expect long runtimes for actual runs.

```bash
# Full run
python tools/run_fpga_flow.py \
  --input-csv experiments/gen1_hapq/ablation/ablation_summary.csv \
  --output-csv experiments/gen1_hapq/fpga_final_results.csv \
  --vivado-mode batch

# Dry run — prints commands only, no execution
python tools/run_fpga_flow.py \
  --input-csv experiments/gen1_hapq/ablation/ablation_summary.csv \
  --dry-run
```

**Outputs:**

| Path | Description |
|---|---|
| `experiments/gen1_hapq/fpga_final_results.csv` | Final comparison table (mAP, LUT, DSP, BRAM, Power, WNS, Bandwidth) |
| `fpga/vivado/eas_snn_hapq/*.rpt` | Raw Vivado reports |

---

## Step 4 — Trade-off Visualization

Plots hardware vs. accuracy trade-off curves from the final FPGA results CSV.

> **Note:** Verify column names in `fpga_final_results.csv` match those expected by the plotting script before running.

```bash
python tools/plot_hapq_tradeoff.py \
  --series \
  Experiments=experiments/gen1_hapq/fpga_final_results.csv \
  --out-png experiments/gen1_hapq/tradeoff_plot.png
```

---

## Step 5 — Compute-Tier Sweep

Simulates HAPQ performance across different edge-device constraints (DSP/BRAM limits). Corresponds to **Section C** of `LaTeX/exp.tex`.

```bash
python tools/sweep_hapq_tiers.py \
  -f exps/default/e_yolox_s_hapq.py \
  -c YOLOX_outputs/e_yolox_s_hapq_target_035/hapq_best_ckpt.pth \
  --tiers Tiny,Small,Medium,Large \
  --result-csv experiments/gen1_hapq/tiers/tier_summary.csv \
  --summary-dir experiments/gen1_hapq/tiers
```

---

## Step 6 — Membrane Potential Sensitivity Sweep

Scans the effect of membrane potential bit-width (`b_U`) on model accuracy, validating the safety of low-bit state quantization. Corresponds to **Fig. A2** of `LaTeX/exp.tex`.

```bash
python tools/run_sensitivity_sweep.py \
  -f exps/default/e_yolox_s_hapq.py \
  -c ckpt/best_ckpt.pth \
  --bits 16,8,6,4 \
  --result-csv experiments/gen1_hapq/sweep_bu/bu_sensitivity.csv \
  --summary-dir experiments/gen1_hapq/sweep_bu \
  --extra-opts --qat-epochs 50 --qat-lr 2e-5 --qat-grad-clip 1.0
```

---

## Appendix — Manual FPGA Debug Flow

Use these steps to debug hardware mapping for a specific configuration.

### 1. Export Verilog Parameters

```bash
python tools/export_hapq_verilog_params.py \
  --summary-json experiments/gen1_hapq/gen1_hapq_summary.json \
  --out-vh fpga/rtl/params.vh
```

### 2. Run Vivado Tcl Script

```bash
vivado -mode batch -source fpga/scripts/synth_hapq.tcl
```

### 3. Parse & Compare Results

```bash
python tools/compare_vivado_cost.py \
  --summary-json experiments/gen1_hapq/gen1_hapq_summary.json \
  --util-rpt fpga/vivado/eas_snn_hapq/utilization_impl.rpt \
  --timing-rpt fpga/vivado/eas_snn_hapq/timing_impl.rpt \
  --power-rpt fpga/vivado/eas_snn_hapq/power.rpt
```

---
