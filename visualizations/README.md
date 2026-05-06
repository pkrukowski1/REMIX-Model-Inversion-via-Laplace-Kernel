# Visualizations

This directory contains the implementation of model inversion pipelines based on pretrained ResNet-34 and ViT-B/16 backbones, together with the plotting utilities used to generate the figures presented throughout the paper.

- `resnet34/` — inversion and visualization pipeline for the ResNet-34 backbone.
- `vit/` — inversion and visualization pipeline for the ViT-B/16 backbone.

In both directories, the `lcm.py` module contains the implementation of the structured LCM-based Negative Log-Likelihood (NLL) computation with efficient $\mathcal{O}(N \log N)$ complexity.