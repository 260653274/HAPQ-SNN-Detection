#!/usr/bin/env bash
set -euo pipefail

#
# EAS-SNN HAPQ FPGA build / simulation wrapper (Vivado batch mode)
#
# Usage:
#   fpga/scripts/run_vivado.sh [synth|sim]
#   - synth (default): 运行综合，输出 utilization/timing 报告
#   - sim:             运行行为仿真
#
# 必须在项目根目录 (EAS-SNN) 下执行，或脚本会自动 cd 到根目录。
#
# Board / Part:
# - Target FPGA part (yours): xczu7ev-ffvc1156-2-i
# - Override: FPGA_PART=xczu7ev-ffvc1156-2-i fpga/scripts/run_vivado.sh
#
# Environment:
# - VIVADO_SETTINGS: path to settings64.sh, e.g.:
#     VIVADO_SETTINGS=/opt/Xilinx/Vivado/2025.2/settings64.sh fpga/scripts/run_vivado.sh
#
# Outputs:
# - synth: fpga/vivado/eas_snn_hapq/utilization_synth.rpt, timing_synth.rpt
# - sim:   控制台输出仿真结果
#

VIVADO_SETTINGS_DEFAULT="/opt/Xilinx/2025.2/Vivado/settings64.sh"

if [[ -n "${VIVADO_SETTINGS:-}" ]]; then
  if [[ ! -f "${VIVADO_SETTINGS}" ]]; then
    echo "VIVADO_SETTINGS points to missing file: ${VIVADO_SETTINGS}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${VIVADO_SETTINGS}"
elif [[ -f "${VIVADO_SETTINGS_DEFAULT}" ]]; then
  # shellcheck disable=SC1090
  source "${VIVADO_SETTINGS_DEFAULT}"
fi

if ! command -v vivado >/dev/null 2>&1; then
  echo "vivado not found in PATH. Source Vivado settings64.sh first." >&2
  echo "Example: source /path/to/Vivado/settings64.sh" >&2
  exit 1
fi

# 确保在项目根目录执行（Cursor 终端可能在不同目录打开）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"
echo "[run_vivado] Working directory: $(pwd)"

MODE="${1:-synth}"
export FPGA_PART="${FPGA_PART:-xczu7ev-ffvc1156-2-i}"
echo "Using FPGA_PART=${FPGA_PART}"
echo "Mode: ${MODE}"

case "${MODE}" in
  sim)
    vivado -mode batch -source fpga/scripts/sim.tcl
    ;;
  synth)
    vivado -mode batch -source fpga/scripts/build.tcl
    ;;
  *)
    echo "Usage: $0 [synth|sim]" >&2
    echo "  synth - 综合 (default)" >&2
    echo "  sim   - 仿真" >&2
    exit 1
    ;;
esac

