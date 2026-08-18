import torch

from task_aware_decorruption.metrics import SegmentationMetrics, psnr


def test_perfect_segmentation() -> None:
    targets = torch.tensor([[[0, 1], [1, 0]]])
    logits = torch.nn.functional.one_hot(targets, num_classes=2).permute(0, 3, 1, 2).float()
    metrics = SegmentationMetrics(2)
    metrics.update(logits, targets)
    result = metrics.compute()
    assert result["miou"] == 1.0
    assert result["pixel_accuracy"] == 1.0


def test_psnr_identity_is_large() -> None:
    image = torch.rand(2, 3, 8, 8)
    assert float(psnr(image, image)) >= 100.0
