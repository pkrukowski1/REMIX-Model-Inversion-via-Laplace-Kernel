# CLIP-Based Continual Learning Implementation

## Baseline Frameworks

This implementation builds upon the following open-source projects:

- [PMI](https://github.com/RuilinTong/PMI-CFS-DFCL) — base framework for data-free continual learning and model inversion.
- [LAMDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT) — continual learning training infrastructure.
- [MoE-Adapters4CL](https://github.com/JiazuoYu/MoE-Adapters4CL) — implementation of the MoE-Adapter architecture used in our CLIP-based experiments.

## Data Preparation

By default, all datasets are expected to be located at:

```bash
/shared/sets/datasets/
```
To modify the dataset location, please edit:
```bash
/utils/data.py
```

## CLIP-Based Continual Learning Experiments
To reproduce the CLIP-based continual learning experiments, run:
```bash
bash ./scripts/moe_adapter_{dataset}.sh
```