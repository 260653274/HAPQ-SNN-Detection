# HAPQ: A Hardware-Aware Pruning and Quantization Pipeline for Event-Based SNN Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Under_Review-blue.svg)](#) 

This repository contains the official implementation of the paper **"HAPQ: A Hardware-Aware Pruning and Quantization Pipeline for Event-Based SNN Detection"** (Currently under review at *MDPI Sensors*).

<p align="center">
  <img src="docs/HAPQ_overview.png" alt="HAPQ Overview" width="85%">
</p>

## 📖 Introduction
The deployment of neuromorphic vision systems on edge devices presents a critical challenge due to irregular execution patterns and substantial temporal state storage overhead. **HAPQ** is a unified hardware-aware co-design pipeline that transcends conventional SNN compression by enforcing "Hardware-Temporal Synergy." 

Key features include:
- **SIMD-Aligned Structured Pruning:** Transforms algorithmic sparsity into hardware-executable regularity.
- **Dual Quantization:** Confines leak factors to a power-of-two form, enabling a 6-bit shift-subtract membrane update.
- **Zero-DSP Deployment:** Achieves 920.81 MHz operating frequency and 0.630 W power consumption on FPGA while improving mAP50:95 to 0.425.

---

## 🙏 Acknowledgments (EAS-SNN)
This project is built upon the excellent foundational architecture of **EAS-SNN**. We would like to express our sincere gratitude to the authors of EAS-SNN for their pioneering work in event-based detection and for making their codebase publicly available. 
* Original EAS-SNN Repository: [https://github.com/chennnng/EAS-SNN](https://github.com/chennnng/EAS-SNN) (Please leave a star for their incredible work!)

---

## 📊 Dataset & Pre-trained Weights

### 1. Prophesee Gen1 Automotive Dataset
We evaluate our HAPQ pipeline comprehensively on the Prophesee Gen1 dataset.
* **Download Link:** [Prophesee Official Gen1 Dataset](https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-dataset/)
* Please follow the official Prophesee guidelines to download and pre-process the event streams into proper formats. Place the processed dataset in the `data/gen1/` directory.

### 2. Pre-trained Weights (EAS-SNN Baseline)
To reproduce our configuration search and fine-tuning, you will need the original EAS-SNN pre-trained weights as the starting point.
* **Download Weights:** [Link to EAS-SNN pre-trained models] *(Note: Please insert your Google Drive / Baidu Pan link or point to the original author's release)*
* Place the downloaded `.pth` files in the `weights/pretrained/` directory.

---

## 🚀 Quick Start & Core Instructions

The core execution scripts for hardware-aware search, pruning, and quantization are located in our dedicated experiment directory. For a complete step-by-step guide, please refer to:
👉 **[`/experiments/gen1_hapq/README.md`](./experiments/gen1_hapq/README.md)**

### Basic Usage Overview
*(Below is a quick summary of the core commands. Please refer to the directory README for detailed hyperparameter configurations.)*

**1. Hardware-Aware Configuration Search:**

```bash
python tools/search.py --config experiments/gen1_hapq/search_config.yaml --dataset gen1
```

**2. SIMD-Aligned Structured Pruning:**

```bash
python tools/prune.py --weights weights/pretrained/eas_snn_baseline.pth --block_size 16 --dsp_budget 0
```

**3. Membrane & Weight Quantization (QAT):**

```bash
python tools/quantize.py --config experiments/gen1_hapq/quant_config.yaml --bit_u 6 --bit_w 8
```

## 📂 Project Structure

```bash
HAPQ-SNN-Detection/
│
├── docs/                      # Image resources and supplementary documents
│   ├── HAPQ_overview.png      # │   └── pruning_strategy.png   # │
├── experiments/gen1_hapq/     # Core instructions, configs, and training scripts for Gen1
│   ├── README.md              # -> Detailed execution instructions <-
│   ├── search_config.yaml
│   └── ...
│
├── models/                    # HAPQ and EAS-SNN model definitions
├── tools/                     # Scripts for training, evaluation, pruning, and quantization
└── README.md                  # This file
```

## 📜 Citation

If you find our HAPQ pipeline or this repository useful in your research, please consider citing our paper **(currently under review)**. We will update this section with the official BibTeX once accepted.

```bash
@article{li2026hapq,
  title={HAPQ: A Hardware-Aware Pruning and Quantization Pipeline for Event-Based SNN Detection},
  author={Li, Zhengyinan and Wu, Jing},
  journal={Submitted to Sensors},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.