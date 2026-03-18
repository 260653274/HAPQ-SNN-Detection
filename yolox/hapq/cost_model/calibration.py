from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class HardwareCalibrationTable:
    dsp_packing: Dict[int, float]
    bram_width_penalty: Dict[int, float]


def default_calibration() -> HardwareCalibrationTable:
    return HardwareCalibrationTable(
        dsp_packing={16: 1.0, 12: 1.0, 8: 2.0, 6: 3.0, 4: 4.0},
        bram_width_penalty={18: 1.0, 36: 1.1, 72: 1.25},
    )
