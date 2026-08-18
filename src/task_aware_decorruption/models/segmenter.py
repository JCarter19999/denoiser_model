import torch
from torch import nn
import torch.nn.functional as F


class SegFormerSegmenter(nn.Module):
    def __init__(self, pretrained_name: str, num_classes: int = 19) -> None:
        super().__init__()
        try:
            from transformers import SegformerForSemanticSegmentation
        except ImportError as exc:
            raise RuntimeError("Install transformers to use SegFormerSegmenter") from exc
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized = (images - self.mean) / self.std
        logits = self.model(pixel_values=normalized).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

    def freeze(self) -> None:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
