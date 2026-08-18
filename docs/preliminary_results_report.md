# Preliminary Results: Task-Aware Image Decorruption

**Project status:** implementation and smoke-test validation complete; full comparative evaluation is pending.

## Objective and approach

This project evaluates whether a lightweight learned image-decorruption front end can improve semantic segmentation of adverse-condition driving scenes. The planned comparison includes raw images, classical enhancement, reconstruction-only decorruption, task-aware decorruption, and a regularized task-aware variant. A SegFormer-B0 segmenter is fine-tuned and then held fixed while the decorrupter is optimized. The full study will report segmentation mIoU (overall, by condition, and by class), image-restoration quality (PSNR and SSIM), and efficiency measures.

The artifacts currently available are intentionally small smoke-test runs. They verify that the data pipeline, model training, checkpointing, reconstruction visualizations, activation recording, and loss-landscape recording run end-to-end. They are useful evidence of a working system, but they are not yet sufficient to claim that task-aware training improves segmentation.

## Results to date

| Component | Configuration | Preliminary result |
| --- | --- | --- |
| Segmentation backbone | SegFormer-B0, Cityscapes-pretrained, 19 classes | After 1 epoch: training loss **0.3749** and validation mIoU **0.4942** |
| Reconstruction-only decorrupter | 24 base channels, residual scale 0.25; synthetic corruption severity 0.25–0.80 | After 1 epoch: composite restoration loss **0.1363** |
| Training records | Fixed reconstruction, activation heatmaps, and local loss landscape | All three MP4 records and final PNG snapshots were produced successfully |

Both smoke tests used 256 × 512 inputs, batch size 1, seed 42, and one training epoch. The restoration objective combines L1 reconstruction loss, SSIM loss (weight 0.25), and smoothness regularization (weight 0.01). The segmenter checkpoint and restoration checkpoint were saved correctly, confirming that the train/evaluate pipeline has reached the point where longer runs and controlled comparisons can be launched.

## Qualitative evidence

The reconstruction animation follows the output for one fixed synthetically corrupted driving image as optimization proceeds. The activation animation provides a complementary view of how encoder/decoder feature responses evolve. Finally, the loss-landscape animation samples a local two-dimensional slice around the current parameters; it is intended as a diagnostic of the nearby optimization geometry rather than a measure of generalization.

> **GIF INSERT 1 — Reconstruction learning**
>
> Insert a GIF converted from `outputs/test_restoration/reconstruction_learning.mp4` here.
>
> *Suggested caption:* “Evolution of the reconstruction-only decorrupter on a fixed synthetic corruption during the one-epoch smoke test.”

> **GIF INSERT 2 — Activation learning**
>
> Insert a GIF converted from `outputs/test_restoration/activation_learning.mp4` here.
>
> *Suggested caption:* “Encoder and decoder activation heatmaps for the fixed reconstruction example during training.”

> **GIF INSERT 3 — Loss-landscape learning**
>
> Insert a GIF converted from `outputs/test_restoration/loss_landscape_learning.mp4` here.
>
> *Suggested caption:* “Recentred local loss-landscape slice through training; this visualization is diagnostic and does not establish model quality by itself.”

Static companion figures are available at `outputs/test_restoration/post_training_reconstruction.png` and `outputs/test_restoration/post_training_loss_landscape.png` if a non-animated version is needed for a PDF or slide deck.

## Interpretation and next steps

The preliminary numbers establish a functioning baseline: the segmentation model reaches 0.4942 validation mIoU in its short test run, and the reconstruction model reduces its configured composite objective to 0.1363. However, no task-aware checkpoint or common-condition evaluation results have been generated yet. Therefore, the central hypothesis—whether optimizing the image front end for the downstream segmentation task outperforms reconstruction-only enhancement—remains untested.

The next experiment should train the segmenter and each decorruption variant to convergence, then evaluate all methods on the same held-out adverse-condition set. The report should then add a method-by-method mIoU table, condition-specific and per-class IoU breakdowns, PSNR/SSIM, and latency/throughput. Qualitative figures should pair each input with its restored image and segmentation prediction, particularly for rain, fog, snow, and nighttime scenes.
