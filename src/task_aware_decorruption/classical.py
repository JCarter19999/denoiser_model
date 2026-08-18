import cv2
import numpy as np
import torch


class ClassicalEnhancer:
    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: int = 8,
        adaptive_gamma: bool = True,
    ) -> None:
        self.clahe = cv2.createCLAHE(
            clipLimit=float(clahe_clip_limit),
            tileGridSize=(int(clahe_grid_size), int(clahe_grid_size)),
        )
        self.adaptive_gamma = adaptive_gamma

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        device, dtype = image.device, image.dtype
        array = (image.detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        lab = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        enhanced = cv2.cvtColor(
            cv2.merge((self.clahe.apply(lightness), a_channel, b_channel)),
            cv2.COLOR_LAB2RGB,
        ).astype(np.float32) / 255.0
        if self.adaptive_gamma:
            mean = float(enhanced.mean())
            if 1e-4 < mean < 0.95:
                gamma = float(np.clip(np.log(0.5) / np.log(mean), 0.5, 2.5))
                enhanced = np.power(np.clip(enhanced, 0, 1), gamma)
        return torch.from_numpy(enhanced).permute(2, 0, 1).to(device=device, dtype=dtype).clamp(0, 1)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 3:
            return self._apply(images)
        if images.ndim != 4:
            raise ValueError("Expected CHW or BCHW tensor")
        return torch.stack([self._apply(image) for image in images])
