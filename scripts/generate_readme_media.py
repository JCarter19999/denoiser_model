import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from task_aware_decorruption.data.clean import CleanImageDataset
from task_aware_decorruption.data.corruptions import _noise
from task_aware_decorruption.models import ResidualDecorrupter
from task_aware_decorruption.utils import load_model_state, resolve_device, seed_everything


def _resize(image: Image.Image, width: int) -> Image.Image:
    if image.width <= width:
        return image
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _save_gif(frames: list[Image.Image], path: Path, duration_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    palette_frames = [
        frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)
        for frame in frames
    ]
    palette_frames[0].save(
        path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def _video_to_gif(video_path: Path, gif_path: Path, *, width: int, max_frames: int) -> None:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = set(np.linspace(0, max(frame_count - 1, 0), min(frame_count, max_frames), dtype=int))
    frames: list[Image.Image] = []
    index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if index in indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(_resize(Image.fromarray(rgb), width))
        index += 1
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames found in {video_path}")
    _save_gif(frames, gif_path, duration_ms=500)


def _to_bgr(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    return cv2.cvtColor(np.round(array.clip(0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _panel(image: torch.Tensor, title: str) -> np.ndarray:
    result = _to_bgr(image)
    cv2.rectangle(result, (0, 0), (result.shape[1], 44), (255, 255, 255), thickness=-1)
    cv2.putText(
        result,
        title,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return result


def _noisy_reconstruction_gif(checkpoint: Path, gif_path: Path) -> None:
    device = resolve_device("auto")
    dataset = CleanImageDataset(
        "data/acdc/rgb_anon",
        (256, 512),
        filename_pattern="*/val_ref/*_rgb_ref_anon.png",
    )
    clean = dataset[0]["image"]
    model = ResidualDecorrupter(base_channels=24, residual_scale=0.25).to(device)
    load_model_state(model, checkpoint)
    model.eval()
    frames: list[Image.Image] = []
    for severity in np.linspace(0.2, 0.95, 8):
        corrupted = _noise(clean, float(severity))
        with torch.no_grad():
            restored = model(corrupted.unsqueeze(0).to(device)).squeeze(0)
        frame = np.concatenate(
            (
                _panel(corrupted, "Noisy input"),
                _panel(restored, "Model reconstruction"),
                _panel(clean, "Clean reference"),
            ),
            axis=1,
        )
        cv2.putText(
            frame,
            f"Synthetic noise severity: {severity:.2f}",
            (12, frame.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        frames.append(_resize(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), 960))
    _save_gif(frames, gif_path, duration_ms=650)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GIF assets for the project README.")
    parser.add_argument("--output-dir", default="assets/results")
    parser.add_argument("--checkpoint", default="outputs/presentation_restoration_final/best.pt")
    args = parser.parse_args()
    seed_everything(42)
    output_dir = Path(args.output_dir)
    video_dir = Path("outputs/presentation_restoration_final")
    _video_to_gif(
        video_dir / "reconstruction_learning.mp4",
        output_dir / "reconstruction_learning.gif",
        width=960,
        max_frames=16,
    )
    _video_to_gif(
        video_dir / "activation_learning.mp4",
        output_dir / "activation_learning.gif",
        width=960,
        max_frames=11,
    )
    _video_to_gif(
        video_dir / "loss_landscape_learning.mp4",
        output_dir / "loss_landscape_learning.gif",
        width=960,
        max_frames=6,
    )
    _noisy_reconstruction_gif(Path(args.checkpoint), output_dir / "noisy_reconstruction.gif")
    print(f"assets={output_dir}")


if __name__ == "__main__":
    main()
