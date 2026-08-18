import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from task_aware_decorruption.config import load_config
from task_aware_decorruption.data import CleanImageDataset, CorruptionPipeline
from task_aware_decorruption.losses import ssim
from task_aware_decorruption.metrics import psnr
from task_aware_decorruption.models import ResidualDecorrupter
from task_aware_decorruption.utils import ensure_dir, load_model_state, resolve_device, seed_everything, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    data_cfg = config["data"]
    bounds = data_cfg.get("corruption_severity", [0.25, 0.8])
    loader = DataLoader(
        CleanImageDataset(
            data_cfg["clean_root"],
            data_cfg["image_size"],
            CorruptionPipeline((float(bounds[0]), float(bounds[1]))),
            data_cfg.get("clean_glob"),
        ),
        batch_size=int(data_cfg.get("batch_size", 4)),
        shuffle=False,
        num_workers=int(data_cfg.get("workers", 4)),
    )
    model_cfg = config["model"]
    model = ResidualDecorrupter(
        base_channels=int(model_cfg.get("base_channels", 24)),
        residual_scale=float(model_cfg.get("residual_scale", 0.25)),
    ).to(device)
    load_model_state(model, args.checkpoint)
    model.eval()

    totals = {"psnr": 0.0, "ssim": 0.0, "l1": 0.0}
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="restoration"):
            clean = batch["clean"].to(device)
            restored = model(batch["corrupted"].to(device))
            batch_size = clean.shape[0]
            totals["psnr"] += float(psnr(restored, clean)) * batch_size
            totals["ssim"] += float(ssim(restored, clean)) * batch_size
            totals["l1"] += float(torch.nn.functional.l1_loss(restored, clean)) * batch_size
            count += batch_size
    results = {key: value / max(1, count) for key, value in totals.items()}
    write_json(ensure_dir(config["output_dir"]) / "restoration.json", results)
    print(results)


if __name__ == "__main__":
    main()
