# DVMeta-DDIE

This repository contains the PyTorch implementation of **Task-Adaptive Dual-View Metric Learning for Few-Shot Drug–Drug Interaction Event Prediction**.

DVMeta-DDIE integrates HyperbolicGIN-based molecular feature extraction, structural association modeling with learnable hypernodes, class prototype calibration, and relation metric learning to predict unseen drug–drug interaction events in few-shot settings.

## Requirements

- Python 3.9.23
- PyTorch 2.3.0
- PyTorch Geometric 2.6.1
- CUDA 11.8
- RDKit 2022.3.3
- NumPy 1.26.0
- pandas 2.2.3

## Installation

```bash
conda create -n dvmeta-ddie python=3.9.23
conda activate dvmeta-ddie
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.6.1 rdkit==2022.3.3 numpy==1.26.0 pandas==2.2.3
```
