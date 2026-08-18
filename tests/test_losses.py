import torch

from task_aware_decorruption.losses import reconstruction_loss, ssim, task_aware_loss


def test_ssim_identity() -> None:
    image = torch.rand(2, 3, 32, 32)
    assert float(ssim(image, image)) > 0.999


def test_reconstruction_loss_is_finite() -> None:
    restored = torch.rand(2, 3, 32, 32, requires_grad=True)
    clean = torch.rand(2, 3, 32, 32)
    loss, parts = reconstruction_loss(restored, clean, l1_weight=1.0, ssim_weight=0.25, smoothness_weight=0.01)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"l1", "ssim", "smoothness"}


def test_task_aware_loss_is_finite() -> None:
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    masks = torch.randint(0, 4, (2, 16, 16))
    restored = torch.rand(2, 3, 16, 16, requires_grad=True)
    adverse = torch.rand(2, 3, 16, 16)
    clean = torch.rand(2, 3, 16, 16)
    clean_restored = torch.rand(2, 3, 16, 16, requires_grad=True)
    loss, parts = task_aware_loss(
        logits,
        masks,
        restored,
        adverse,
        clean_identity=clean,
        clean_restored=clean_restored,
        ignore_index=255,
        segmentation_weight=1.0,
        identity_weight=0.1,
        smoothness_weight=0.01,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"segmentation", "identity", "smoothness"}
