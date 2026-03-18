# Gen1 HAPQ 闭环实验关键 Terminal 指令

本文件仅保留“按顺序可执行”的核心命令，覆盖训练-搜索-消融-FPGA实现-对比分析全流程。

当前约定：
- Conda 环境：`eas-snn`
- FPGA 器件：`xczu7ev-ffvc1156-2-i` (或根据实际情况调整)
- 主板连接：JTAG + UART

## 0) 环境准备（每次新终端先执行）

```bash
cd /home/xlang/EAS-SNN
source ~/.bashrc
conda activate eas-snn

# 可选：检查 CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 1) 训练 + 搜索 + QAT（生成 HAPQ 总结文件）

```bash
# 优化策略(独立保存，不覆盖旧结果):
# 新增参数: -expn e_yolox_s_hapq_target_035
# 这将在 YOLOX_outputs/ 下创建一个名为 e_yolox_s_hapq_target_035 的新文件夹

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

产物(HAPQ最佳配置)：
- `experiments/gen1_hapq/gen1_hapq_summary_04.json` (HAPQ 最佳配置)
- `YOLOX_outputs/e_yolox_s_hapq_target_035/hapq_best_ckpt.pth`

## 2) 软件消融实验 (Ablation Study)

生成 Prune-only, Quant-W, Quant-W+U 等变体并获取关键指标 (mAP50:95, mAP50, Params, SynOps)。
实验设计参考 LaTeX/exp.tex Section A2 (Metrics) 和 A3 (Ablation Grid)。

**消融维度说明：**
1. **Is Structure Necessary?** (Pruning Granularity): 验证结构化剪枝的有效性。
2. **Is State Quantization Safe?** (Membrane Precision): 对比 Quant-W (FP16 State) vs. Quant-W+U (INT8 State)。
3. **Is NAS Effective?** (Search Strategy): 验证混合精度搜索优于固定策略。

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

```bash
# 快速重新评估指标 (Eval-only, 不重新训练):
# 前提：之前已经运行过上述命令并生成了 checkpoints
# 新增 "hapq" 模式以包含主实验结果 (YOLOX_outputs/e_yolox_s_hapq_target_035)
python tools/run_ablation.py \
  -f exps/default/e_yolox_s_hapq.py \
  --ckpt ckpt/best_ckpt.pth \
  --experiment-prefix gen1_ablation \
  --modes prune_only,quant_w,quant_wu,full,hapq \
  --result-csv experiments/gen1_hapq/ablation/ablation_summary.csv \
  --summary-dir experiments/gen1_hapq/ablation \
  --extra-opts --qat-epochs 0
```

产物：
- `experiments/gen1_hapq/ablation/ablation_summary.csv`: 包含 mAP, mAP50, Params, SynOps, KeepRatio 等指标的汇总表。
- `experiments/gen1_hapq/ablation/*.json`: 各变体的详细配置和指标。

## 3) FPGA 硬件对比实验 (Automated Flow)

### 3.1 生成 Baseline 配置 (SNN-FP16/No-Pruning)

```bash
python tools/gen_fpga_baseline.py \
  --template-json experiments/gen1_hapq/gen1_hapq_summary_04.json \
  --out-json experiments/gen1_hapq/ablation/baseline_summary.json \
  --bitwidth 16
```

### 3.2 手动添加 Baseline 到消融列表 (可选)

如果需要将 Baseline 也纳入统一流程，可将其添加到 `ablation_summary.csv`，或单独运行。
为了方便，我们可以手动创建一个包含所有待测配置的列表，或者直接运行下一步（流程脚本支持跳过缺失项）。

### 3.3 运行 FPGA 综合与实现 (Batch Mode)

此步骤自动执行：导出 Verilog 参数 -> Vivado 综合/实现 -> 解析资源/功耗/带宽 -> 汇总结果。

**前提**：确保 `vivado` 命令在 PATH 中，或使用 Dry-Run 模式测试。

```bash
# 实际运行 (耗时较长)
python tools/run_fpga_flow.py \
  --input-csv experiments/gen1_hapq/ablation/ablation_summary.csv \
  --output-csv experiments/gen1_hapq/fpga_final_results.csv \
  --vivado-mode batch

# Dry-Run (仅打印命令，用于检查流程)
python tools/run_fpga_flow.py \
  --input-csv experiments/gen1_hapq/ablation/ablation_summary.csv \
  --dry-run
```

产物：
- `experiments/gen1_hapq/fpga_final_results.csv`: 包含 mAP, LUT, DSP, BRAM, Power, WNS, Bandwidth 的最终对比表。
- `fpga/vivado/eas_snn_hapq/*.rpt`: 原始 Vivado 报告。

## 4) 可视化 (Trade-off Analysis)

使用最终生成的硬件指标 CSV 绘制图表。

```bash
# 注意：需根据 fpga_final_results.csv 的实际列名调整绘图脚本，或使用以下通用绘图
python tools/plot_hapq_tradeoff.py \
  --series \
  Experiments=experiments/gen1_hapq/fpga_final_results.csv \
  --out-png experiments/gen1_hapq/tradeoff_plot.png
```

## 5) 多层级计算资源搜索 (Compute Tiers Sweep)

对应 LaTeX/exp.tex Section C，模拟不同边缘端约束 (DSP/BRAM limits) 下的 HAPQ 性能。

```bash
python tools/sweep_hapq_tiers.py \
  -f exps/default/e_yolox_s_hapq.py \
  -c YOLOX_outputs/e_yolox_s_hapq_target_035/hapq_best_ckpt.pth \
  --tiers Tiny,Small,Medium,Large \
  --result-csv experiments/gen1_hapq/tiers/tier_summary.csv \
  --summary-dir experiments/gen1_hapq/tiers
```

## 6) 补充实验：膜电位量化敏感度扫描 (Sensitivity Sweep)

对应 LaTeX/exp.tex Fig-A2，扫描不同膜电位位宽 (b_U) 对精度的影响，验证低比特状态量化的安全性。

```bash
python tools/run_sensitivity_sweep.py \
  -f exps/default/e_yolox_s_hapq.py \
  -c ckpt/best_ckpt.pth \
  --bits 16,8,6,4 \
  --result-csv experiments/gen1_hapq/sweep_bu/bu_sensitivity.csv \
  --summary-dir experiments/gen1_hapq/sweep_bu \
  --extra-opts --qat-epochs 50 --qat-lr 2e-5 --qat-grad-clip 1.0
```

## 附录：手动单次运行 FPGA 流程 (调试用)

如果需要调试特定配置的硬件映射：

1. **导出参数**
   ```bash
   python tools/export_hapq_verilog_params.py \
     --summary-json experiments/gen1_hapq/gen1_hapq_summary.json \
     --out-vh fpga/rtl/params.vh
   ```

2. **运行 Vivado Tcl**
   ```bash
   vivado -mode batch -source fpga/scripts/synth_hapq.tcl
   ```

3. **解析结果**
   ```bash
   python tools/compare_vivado_cost.py \
     --summary-json experiments/gen1_hapq/gen1_hapq_summary.json \
     --util-rpt fpga/vivado/eas_snn_hapq/utilization_impl.rpt \
     --timing-rpt fpga/vivado/eas_snn_hapq/timing_impl.rpt \
     --power-rpt fpga/vivado/eas_snn_hapq/power.rpt
   ```
