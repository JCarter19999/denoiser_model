import torch

from task_aware_decorruption.data.corruptions import CorruptionPipeline
from task_aware_decorruption.losses import reconstruction_loss, task_aware_loss
from task_aware_decorruption.metrics import SegmentationMetrics
from task_aware_decorruption.models.decorrupter import ResidualDecorrupter


def main() -> None:
    torch.manual_seed(7)
    images = torch.rand(2, 3, 64, 96)
    pipeline = CorruptionPipeline((0.4, 0.4))
    corrupted = torch.stack([pipeline(image)[0] for image in images])
    model = ResidualDecorrupter(base_channels=8)
    restored = model(corrupted)
    restoration, _ = reconstruction_loss(
        restored,
        images,
        l1_weight=1.0,
        ssim_weight=0.25,
        smoothness_weight=0.01,
    )
    logits = torch.randn(2, 4, 64, 96, requires_grad=True)
    masks = torch.randint(0, 4, (2, 64, 96))
    task_loss, _ = task_aware_loss(
        logits,
        masks,
        restored,
        corrupted,
        clean_identity=images,
        clean_restored=model(images),
        ignore_index=255,
        segmentation_weight=1.0,
        identity_weight=0.1,
        smoothness_weight=0.01,
    )
    (restoration + task_loss).backward()
    metrics = SegmentationMetrics(4)
    metrics.update(logits.detach(), masks)
    result = metrics.compute()
    assert restored.shape == images.shape
    assert 0.0 <= float(result["miou"]) <= 1.0
    assert any(parameter.grad is not None for parameter in model.parameters())
    print("Smoke test passed")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")


if __name__ == "__main__":
    main()
