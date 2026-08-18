# Experiment Design

The segmenter is fine-tuned once and frozen during decorrupter training.

Primary comparisons:

1. Raw adverse image
2. Classical enhancement
3. Reconstruction-only decorrupter
4. Task-aware decorrupter
5. Task-aware decorrupter with regularization

Primary metrics are overall mIoU, condition-specific mIoU, per-class IoU, PSNR, SSIM, latency, throughput, parameters, and peak GPU memory.
