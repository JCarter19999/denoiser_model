import cv2
import numpy as np
import torch
from pathlib import Path


def _to_bgr(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    return cv2.cvtColor(np.round(array.clip(0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _labeled(
    image: np.ndarray,
    label: str,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    result = image.copy()
    cv2.putText(
        result,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )
    return result


def show_reconstruction_preview(
    corrupted: torch.Tensor,
    restored: torch.Tensor,
    clean: torch.Tensor,
    status: str,
) -> bool:
    frame = reconstruction_frame(corrupted, restored, clean, status)
    cv2.imshow("Training reconstruction preview (q to stop preview)", frame)
    return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)


def reconstruction_frame(
    corrupted: torch.Tensor,
    restored: torch.Tensor,
    clean: torch.Tensor,
    status: str,
) -> np.ndarray:
    frame = np.concatenate(
        (
            _labeled(_to_bgr(corrupted), "Fixed corruption"),
            _labeled(_to_bgr(restored), "Current reconstruction", (40, 40, 40)),
            _labeled(_to_bgr(clean), "Clean target", (40, 40, 40)),
        ),
        axis=1,
    )
    cv2.putText(
        frame,
        status,
        (12, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return frame


class VideoRecorder:
    def __init__(self, path: str | Path, fps: float = 3.0) -> None:
        self.path = Path(path)
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (width, height),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not create video: {self.path}")
        self.writer.write(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


class ActivationCapture:
    layer_names = ("stem", "down1", "down2", "down3", "up2", "up1", "up0")

    def __init__(self, model: torch.nn.Module) -> None:
        self.activations: dict[str, torch.Tensor] = {}
        self.handles = [
            getattr(model, name).register_forward_hook(self._capture(name))
            for name in self.layer_names
        ]

    def _capture(self, name: str):
        def hook(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.activations[name] = output.detach()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _activation_tile(activation: torch.Tensor, label: str, size: tuple[int, int]) -> np.ndarray:
    values = activation.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()
    normalized = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    heatmap = cv2.resize(heatmap, size, interpolation=cv2.INTER_NEAREST)
    return _labeled(heatmap, label, (30, 30, 30))


def activation_dashboard_frame(
    corrupted: torch.Tensor,
    restored: torch.Tensor,
    activations: dict[str, torch.Tensor],
    status: str,
) -> np.ndarray:
    height, width = corrupted.shape[-2:]
    tile_size = (width, height)
    tiles = [
        _labeled(_to_bgr(corrupted), "Fixed corruption"),
        _labeled(_to_bgr(restored), "Current reconstruction", (40, 40, 40)),
    ]
    tiles.extend(
        _activation_tile(activations[name], name, tile_size)
        for name in ActivationCapture.layer_names
    )
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[index : index + 3], axis=1) for index in range(0, len(tiles), 3)]
    frame = np.concatenate(rows, axis=0)
    cv2.putText(
        frame,
        status,
        (12, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return frame


def close_reconstruction_preview() -> None:
    cv2.destroyWindow("Training reconstruction preview (q to stop preview)")
