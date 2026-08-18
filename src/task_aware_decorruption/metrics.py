import math

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        if predictions.ndim == 4:
            predictions = predictions.argmax(dim=1)
        predictions = predictions.detach().cpu().long().reshape(-1)
        targets = targets.detach().cpu().long().reshape(-1)
        valid = (targets != self.ignore_index) & (targets >= 0) & (targets < self.num_classes)
        encoded = targets[valid] * self.num_classes + predictions[valid]
        counts = torch.bincount(encoded, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        matrix = self.confusion.float()
        intersection = matrix.diag()
        union = matrix.sum(1) + matrix.sum(0) - intersection
        iou = torch.where(union > 0, intersection / union, torch.nan)
        miou = torch.nanmean(iou).item()
        accuracy = intersection.sum().div(matrix.sum().clamp_min(1)).item()
        return {
            "miou": miou,
            "pixel_accuracy": accuracy,
            "per_class_iou": [None if math.isnan(value) else value for value in iou.tolist()],
            "confusion_matrix": self.confusion.tolist(),
        }


def psnr(restored: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((restored - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return (10.0 * torch.log10(1.0 / mse)).mean()
