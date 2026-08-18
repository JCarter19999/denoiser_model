import math
import random

import torch
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    axis = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(axis**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)


def _blur(image: torch.Tensor, severity: float) -> torch.Tensor:
    size = 3 + 2 * int(round(severity * 3))
    sigma = 0.5 + 2.0 * severity
    kernel = _gaussian_kernel(size, sigma, image.device, image.dtype)
    weight = kernel.expand(image.shape[0], 1, size, size)
    return F.conv2d(image.unsqueeze(0), weight, padding=size // 2, groups=image.shape[0]).squeeze(0)


def _noise(image: torch.Tensor, severity: float) -> torch.Tensor:
    return (image + torch.randn_like(image) * (0.02 + 0.10 * severity)).clamp(0, 1)


def _darken(image: torch.Tensor, severity: float) -> torch.Tensor:
    gamma = 1.0 + 2.5 * severity
    return image.clamp(0, 1).pow(gamma)


def _contrast(image: torch.Tensor, severity: float) -> torch.Tensor:
    factor = 1.0 - 0.75 * severity
    mean = image.mean(dim=(-2, -1), keepdim=True)
    return ((image - mean) * factor + mean).clamp(0, 1)


def _fog(image: torch.Tensor, severity: float) -> torch.Tensor:
    h, w = image.shape[-2:]
    low_h = max(2, h // 32)
    low_w = max(2, w // 32)
    haze = torch.rand(1, 1, low_h, low_w, device=image.device, dtype=image.dtype)
    haze = F.interpolate(haze, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
    haze = _blur(haze.expand_as(image), min(1.0, severity + 0.2))
    strength = 0.25 + 0.45 * severity
    veil = (0.75 + 0.25 * haze).clamp(0, 1)
    return (image * (1.0 - strength) + veil * strength).clamp(0, 1)


def _rain(image: torch.Tensor, severity: float) -> torch.Tensor:
    c, h, w = image.shape
    rain = torch.zeros((1, h, w), device=image.device, dtype=image.dtype)
    count = int((80 + 320 * severity) * h * w / (384 * 768))
    length = min(h, max(1, int(8 + 18 * severity)))

    for _ in range(max(1, count)):
        y = random.randrange(h - length + 1)
        x = random.randrange(w)
        for step in range(length):
            xx = min(w - 1, x + step // 4)
            rain[0, y + step, xx] = 1.0
    rain = _blur(rain.expand(c, -1, -1), 0.15)
    return (image + rain * (0.10 + 0.20 * severity)).clamp(0, 1)


def _snow(image: torch.Tensor, severity: float) -> torch.Tensor:
    c, h, w = image.shape
    probability = 0.003 + 0.018 * severity
    flakes = (torch.rand((1, h, w), device=image.device) < probability).to(image.dtype)
    flakes = _blur(flakes.expand(c, -1, -1), 0.3 + 0.4 * severity)
    return (image + flakes * (0.35 + 0.55 * severity)).clamp(0, 1)


class CorruptionPipeline:
    names = ("blur", "noise", "darken", "contrast", "fog", "rain", "snow")

    def __init__(self, severity: tuple[float, float] = (0.25, 0.8)) -> None:
        self.low = float(severity[0])
        self.high = float(severity[1])

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, str, float]:
        name = random.choice(self.names)
        level = random.uniform(self.low, self.high)
        functions = {
            "blur": _blur,
            "noise": _noise,
            "darken": _darken,
            "contrast": _contrast,
            "fog": _fog,
            "rain": _rain,
            "snow": _snow,
        }
        return functions[name](image, level), name, level
