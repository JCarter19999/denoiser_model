import argparse
from pathlib import Path

import numpy as np
import torch

from task_aware_decorruption.config import load_config
from task_aware_decorruption.data import CleanImageDataset, CorruptionPipeline
from task_aware_decorruption.loss_landscape import compute_local_loss_landscape, save_loss_landscape
from task_aware_decorruption.models import ResidualDecorrupter
from task_aware_decorruption.utils import (
    ensure_dir,
    load_model_state,
    resolve_device,
    seed_everything,
)


def _fixed_batch(
    dataset: CleanImageDataset,
    samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    items = [dataset[index] for index in range(samples)]
    corrupted = torch.stack([item["corrupted"] for item in items]).to(device)
    clean = torch.stack([item["clean"] for item in items]).to(device)
    return corrupted, clean


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a local reconstruction-loss landscape around a checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/loss_landscape.png")
    parser.add_argument("--resolution", type=int, default=21)
    parser.add_argument("--span", type=float, default=0.25)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()
    if args.resolution < 3:
        parser.error("--resolution must be at least 3")
    if args.span <= 0:
        parser.error("--span must be positive")
    if args.samples < 1:
        parser.error("--samples must be positive")

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
    corrupted, clean = _fixed_batch(dataset, min(args.samples, len(dataset)), device)
    model_config = config["model"]
    model = ResidualDecorrupter(
        base_channels=int(model_config.get("base_channels", 24)),
        residual_scale=float(model_config.get("residual_scale", 0.25)),
    ).to(device)
    load_model_state(model, args.checkpoint)
    model.eval()

    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    landscape = compute_local_loss_landscape(
        model,
        corrupted,
        clean,
        config["loss"],
        resolution=args.resolution,
        span=args.span,
    )
    save_loss_landscape(landscape, "Local Reconstruction-Loss Surface", output_path)
    minimum = np.unravel_index(landscape.values.argmin(), landscape.values.shape)
    maximum = np.unravel_index(landscape.values.argmax(), landscape.values.shape)
    print(f"saved={output_path}")
    print(
        f"global_minimum={landscape.values[minimum]:.6f} "
        f"at ({landscape.coordinates[minimum[1]]:.3f}, {landscape.coordinates[minimum[0]]:.3f})"
    )
    print(
        f"global_maximum={landscape.values[maximum]:.6f} "
        f"at ({landscape.coordinates[maximum[1]]:.3f}, {landscape.coordinates[maximum[0]]:.3f})"
    )


if __name__ == "__main__":
    main()
