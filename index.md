---
title: Compressed Sensing for MRI
description: An interactive online course on compressed sensing and its application to rapid MR imaging
---

Compressed sensing allows MRI scanners to acquire far fewer k-space samples than the Nyquist limit requires and still reconstruct faithful images — meaning faster scans with less time in the scanner. This course builds the intuition behind compressed sensing from the ground up, using interactive figures you can explore directly in the browser.

## Chapters

**[Intro to wavelets](content/00_wavelet_intro.ipynb)** — Build a 1D wavelet transform from scratch through a discovery-based activity. You will start with kernels that fail, identify why they fail, and derive each wavelet property from necessity — arriving at a multi-scale decomposition that concentrates signal energy into a small number of coefficients.

**[2D Wavelet Decomposition](content/01_2d_wavelet_primer.ipynb)** — Extend wavelets to two dimensions with a brief visual overview. See how the 1D filter bank applies separably to images (rows then columns), how the cascade recursively decomposes coarser features, and how thresholding small coefficients compresses an image — the same principle behind JPEG 2000.

**[1D Compressed Sensing](content/03_fig2_demo.ipynb)** — Discover why *random* undersampling enables signal recovery while *uniform* undersampling does not. Random sampling spreads aliasing into noise-like interference that thresholding can remove; uniform sampling creates coherent ghosts that overlap with the true signal. This chapter reproduces the intuitive 1D demonstration from Lustig et al. (2007), Figure 2.

**[2D MRI Compressed Sensing](content/04_fig3_demo.ipynb)** — See the full compressed sensing pipeline applied to a real MRI brain slice. Three domains are at play: k-space (where we sample), the image domain (what we want), and the wavelet domain (where the image is sparse). An iterative reconstruction alternates between enforcing sparsity in the wavelet domain and data consistency in k-space, progressively sharpening the image. Inspired by Lustig et al. (2007).

## Reference

> Lustig M, Donoho D, Pauly JM. **Sparse MRI: The Application of Compressed Sensing for Rapid MR Imaging.**
> *Magnetic Resonance in Medicine*, 58(6):1182–1195, 2007.
