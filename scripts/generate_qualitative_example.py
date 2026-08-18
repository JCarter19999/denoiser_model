import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from task_aware_decorruption.data.clean import CleanImageDataset
from task_aware_decorruption.data.corruptions import _darken, _fog, _noise, _rain
from task_aware_decorruption.models import ResidualDecorrupter
from task_aware_decorruption.utils import load_model_state, resolve_device, seed_everything


def _to_bgr(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    return cv2.cvtColor(np.round(array.clip(0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _panel(image: torch.Tensor, title: str, color: tuple[int, int, int]) -> np.ndarray:
    result = _to_bgr(image)
    cv2.rectangle(result, (0, 0), (result.shape[1], 52), (255, 255, 255), thickness=-1)
    cv2.putText(result, title, (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a severe qualitative restoration example."
    )
    parser.add_argument("--checkpoint", default="outputs/presentation_restoration_final/best.pt")
    parser.add_argument(
        "--output",
        default="outputs/presentation_restoration_final/qualitative_severe_reconstruction.png",
    )
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    seed_everything(42)
    device = resolve_device("auto")
    dataset = CleanImageDataset(
        "data/acdc/rgb_anon",
        (384, 768),
        filename_pattern="*/val_ref/*_rgb_ref_anon.png",
    )
    clean = dataset[args.index % len(dataset)]["image"]
    corrupted = _darken(clean, 0.8)
    corrupted = _fog(corrupted, 0.85)
    corrupted = _rain(corrupted, 0.9)
    corrupted = _noise(corrupted, 0.8)
    model = ResidualDecorrupter(base_channels=24, residual_scale=0.25).to(device)
    load_model_state(model, args.checkpoint)
    model.eval()
    with torch.no_grad():
        restored = model(corrupted.unsqueeze(0).to(device)).squeeze(0)
    figure = np.concatenate(
        (
            _panel(corrupted, "Severe composite corruption", (50, 50, 50)),
            _panel(restored, "Model reconstruction", (50, 50, 50)),
            _panel(clean, "Clean reference", (50, 50, 50)),
        ),
        axis=1,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), figure)
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
