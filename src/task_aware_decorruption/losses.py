import torch
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(axis**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    window = torch.outer(kernel, kernel)
    return window.expand(channels, 1, size, size)


def ssim(image_a: torch.Tensor, image_b: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    channels = image_a.shape[1]
    window = _gaussian_window(window_size, 1.5, channels, image_a.device, image_a.dtype)
    padding = window_size // 2
    mu_a = F.conv2d(image_a, window, padding=padding, groups=channels)
    mu_b = F.conv2d(image_b, window, padding=padding, groups=channels)
    sigma_a = F.conv2d(image_a * image_a, window, padding=padding, groups=channels) - mu_a**2
    sigma_b = F.conv2d(image_b * image_b, window, padding=padding, groups=channels) - mu_b**2
    sigma_ab = F.conv2d(image_a * image_b, window, padding=padding, groups=channels) - mu_a * mu_b
    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    )
    return score.mean()


def total_variation(image: torch.Tensor) -> torch.Tensor:
    vertical = (image[:, :, 1:] - image[:, :, :-1]).abs().mean()
    horizontal = (image[:, :, :, 1:] - image[:, :, :, :-1]).abs().mean()
    return vertical + horizontal


def reconstruction_loss(
    restored: torch.Tensor,
    clean: torch.Tensor,
    *,
    l1_weight: float,
    ssim_weight: float,
    smoothness_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.l1_loss(restored, clean)
    ssim_loss = 1.0 - ssim(restored, clean)
    smoothness = total_variation(restored - clean)
    total = l1_weight * l1 + ssim_weight * ssim_loss + smoothness_weight * smoothness
    return total, {"l1": l1, "ssim": ssim_loss, "smoothness": smoothness}


def task_aware_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    restored: torch.Tensor,
    adverse: torch.Tensor,
    *,
    clean_identity: torch.Tensor,
    clean_restored: torch.Tensor,
    ignore_index: int,
    segmentation_weight: float,
    identity_weight: float,
    smoothness_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    segmentation = F.cross_entropy(logits, masks, ignore_index=ignore_index)
    identity = F.l1_loss(clean_restored, clean_identity)
    smoothness = total_variation(restored - adverse)
    total = segmentation_weight * segmentation + identity_weight * identity + smoothness_weight * smoothness
    return total, {"segmentation": segmentation, "identity": identity, "smoothness": smoothness}
