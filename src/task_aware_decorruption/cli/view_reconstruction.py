import argparse

import cv2
import numpy as np
import torch

from task_aware_decorruption.config import load_config
from task_aware_decorruption.data import CleanImageDataset, CorruptionPipeline
from task_aware_decorruption.models import ResidualDecorrupter
from task_aware_decorruption.utils import load_model_state, resolve_device, seed_everything


def _to_bgr(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().permute(1, 2, 0).numpy()
    return cv2.cvtColor(np.round(array.clip(0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def _labeled(image: np.ndarray, label: str) -> np.ndarray:
    result = image.copy()
    cv2.putText(
        result,
        label,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show live reconstruction results from a decorrupter checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--delay-ms", type=int, default=250)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    data_config = config["data"]
    bounds = data_config.get("corruption_severity", [0.25, 0.8])
    dataset = CleanImageDataset(
        data_config["clean_root"],
        data_config["image_size"],
        CorruptionPipeline((float(bounds[0]), float(bounds[1]))),
        data_config.get("clean_glob"),
    )
    model_config = config["model"]
    model = ResidualDecorrupter(
        base_channels=int(model_config.get("base_channels", 24)),
        residual_scale=float(model_config.get("residual_scale", 0.25)),
    ).to(device)
    load_model_state(model, args.checkpoint)
    model.eval()

    index = 0
    while True:
        item = dataset[index]
        corrupted = item["corrupted"].unsqueeze(0).to(device)
        with torch.no_grad():
            restored = model(corrupted).squeeze(0)
        frame = np.concatenate(
            (
                _labeled(_to_bgr(item["corrupted"]), "Corrupted"),
                _labeled(_to_bgr(restored), "Restored"),
                _labeled(_to_bgr(item["clean"]), "Clean target"),
            ),
            axis=1,
        )
        cv2.imshow("Decorrupter reconstruction (q to quit)", frame)
        key = cv2.waitKey(max(1, args.delay_ms)) & 0xFF
        if key in (ord("q"), 27):
            break
        index += 1
        if index == len(dataset):
            if not args.loop:
                break
            index = 0

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
