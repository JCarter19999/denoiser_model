import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class ResidualDecorrupter(nn.Module):
    def __init__(self, base_channels: int = 24, residual_scale: float = 0.25) -> None:
        super().__init__()
        b = base_channels
        self.residual_scale = residual_scale
        self.stem = ConvBlock(3, b)
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)
        self.down3 = DownBlock(b * 4, b * 8)
        self.up2 = UpBlock(b * 8, b * 4, b * 4)
        self.up1 = UpBlock(b * 4, b * 2, b * 2)
        self.up0 = UpBlock(b * 2, b, b)
        self.head = nn.Conv2d(b, 3, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x0 = self.stem(image)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)
        residual = torch.tanh(self.head(x)) * self.residual_scale
        return (image + residual).clamp(0.0, 1.0)
