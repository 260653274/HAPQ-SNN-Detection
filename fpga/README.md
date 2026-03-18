# FPGA RTL (综合 + 仿真)

本目录与 Python 训练代码独立，支持 Vivado 综合与行为仿真。

## 目录结构

- `rtl/`: Verilog 模块（block_gate, quantizer, membrane_update, layer_pipeline, snn_top 等）
- `scripts/build.tcl`: Vivado 综合脚本
- `scripts/sim.tcl`: Vivado 仿真脚本
- `scripts/run_vivado.sh`: 统一入口（支持 synth / sim）

## Cursor + Vivado 联调

在项目根目录 (EAS-SNN) 下执行，可从 Cursor 终端直接运行：

```bash
# 仿真（验证 RTL 行为）
fpga/scripts/run_vivado.sh sim

# 综合（生成 utilization/timing 报告）
fpga/scripts/run_vivado.sh synth
```

### 环境要求

若 Vivado 未在 PATH 中，需指定 `VIVADO_SETTINGS`：

```bash
VIVADO_SETTINGS=/opt/Xilinx/2025.2/Vivado/settings64.sh fpga/scripts/run_vivado.sh sim
```

默认会尝试 `/opt/Xilinx/2025.2/Vivado/settings64.sh`，若已 source 或安装路径不同，可覆盖。

### 器件

默认 `FPGA_PART=xczu7ev-ffvc1156-2-i`，可覆盖：

```bash
FPGA_PART=xczu3eg-sbva484-1-e fpga/scripts/run_vivado.sh synth
```

## Outputs

- 综合：`fpga/vivado/eas_snn_hapq/utilization_synth.rpt`、`timing_synth.rpt`
- 仿真：控制台输出，tb 中 `$display` 可见
