# Differentiable GPU-friendly Direct Linear Transform
This repository contains the PyTorch implementation of the GPU-friendly formulation of Direct Linear Transform proposed in the paper ["Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation"](https://arxiv.org/abs/2004.02186) (CVPR 2020).

The Direct Linear Transform (DLT) algorithm can be used to solve a set of similarity equations of the form

x<sub>k</sub> = &alpha; A y<sub>k</sub> for k in range(K),

where &alpha; is an uknown scalar mulitplier.

In our paper, we make use of DLT to lift a set of 2D detections together with the associated projection matrices to 3D positions in a differentiable fashion.
To do so efficiently and use this operation to supervise training of our multi-view pose estimation pipeline, we propose a novel implementation of DLT based on Shifted Inverse Iterations (SII) that is orders of magnitude faster than standard SVD-based ones on GPU architectures.

# Getting started
This module doesn't have any difficult-to-install dependencies. To get started and install its minimal dependencies simply run:
```
git clone 
./setup.sh
```
