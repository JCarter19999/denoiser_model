import argparse
import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from task_aware_decorruption.config import load_config
from task_aware_decorruption.data.acdc import ACDCDataset
from task_aware_decorruption.engine import evaluate_segmentation
from task_aware_decorruption.models.segmenter import SegFormerSegmenter
from task_aware_decorruption.utils import ensure_dir, resolve_device, save_checkpoint, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["output_dir"])
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    train_loader = DataLoader(
        ACDCDataset(data_cfg["root"], data_cfg["train_split"], data_cfg["image_size"]),
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("workers", 4)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        ACDCDataset(data_cfg["root"], data_cfg["val_split"], data_cfg["image_size"]),
        batch_size=1,
        shuffle=False,
        num_workers=int(data_cfg.get("workers", 4)),
        pin_memory=device.type == "cuda",
    )

    model = SegFormerSegmenter(model_cfg["pretrained_name"], int(model_cfg.get("num_classes", 19))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = -math.inf
    bad_epochs = 0

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"segmenter {epoch}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = F.cross_entropy(
                    model(images),
                    masks,
                    ignore_index=int(data_cfg.get("ignore_index", 255)),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        overall, _ = evaluate_segmentation(
            model,
            val_loader,
            device,
            num_classes=int(model_cfg.get("num_classes", 19)),
            ignore_index=int(data_cfg.get("ignore_index", 255)),
            description="validate",
        )
        miou = float(overall["miou"])
        metrics = {"train_loss": running / max(1, len(train_loader)), "miou": miou}
        save_checkpoint(output_dir / "last.pt", model=model, optimizer=optimizer, epoch=epoch, metrics=metrics, config=config)
        if miou > best:
            best = miou
            bad_epochs = 0
            save_checkpoint(output_dir / "best.pt", model=model, optimizer=optimizer, epoch=epoch, metrics=metrics, config=config)
        else:
            bad_epochs += 1
            if bad_epochs >= int(train_cfg.get("patience", 5)):
                break


if __name__ == "__main__":
    main()
