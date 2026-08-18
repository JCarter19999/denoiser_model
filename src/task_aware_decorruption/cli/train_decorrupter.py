import argparse
from itertools import cycle

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from task_aware_decorruption.config import load_config
from task_aware_decorruption.data import ACDCDataset, CleanImageDataset, CorruptionPipeline
from task_aware_decorruption.losses import reconstruction_loss, task_aware_loss
from task_aware_decorruption.loss_landscape import (
    compute_local_loss_landscape,
    normalized_directions,
    render_loss_landscape,
    save_loss_landscape,
)
from task_aware_decorruption.models import ResidualDecorrupter, SegFormerSegmenter
from task_aware_decorruption.utils import (
    ensure_dir,
    load_model_state,
    resolve_device,
    save_checkpoint,
    seed_everything,
)
from task_aware_decorruption.visualization import (
    ActivationCapture,
    VideoRecorder,
    activation_dashboard_frame,
    close_reconstruction_preview,
    reconstruction_frame,
    show_reconstruction_preview,
)


def clean_loader(config: dict, device: torch.device, synthetic: bool) -> DataLoader:
    data_cfg = config["data"]
    corruption = None
    if synthetic:
        bounds = data_cfg.get("corruption_severity", [0.25, 0.8])
        corruption = CorruptionPipeline((float(bounds[0]), float(bounds[1])))
    dataset = CleanImageDataset(
        data_cfg["clean_root"],
        data_cfg["image_size"],
        corruption,
        data_cfg.get("clean_glob"),
    )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("workers", 4)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-every", type=int, default=50)
    parser.add_argument("--record-learning", action="store_true")
    parser.add_argument("--record-every", type=int, default=25)
    parser.add_argument("--landscape-every", type=int, default=200)
    parser.add_argument("--landscape-resolution", type=int, default=9)
    parser.add_argument("--landscape-span", type=float, default=0.25)
    parser.add_argument("--record-activations", action="store_true")
    parser.add_argument("--activation-every", type=int, default=100)
    args = parser.parse_args()
    if args.preview_every < 1:
        parser.error("--preview-every must be positive")
    if args.record_every < 1 or args.landscape_every < 1:
        parser.error("--record-every and --landscape-every must be positive")
    if args.activation_every < 1:
        parser.error("--activation-every must be positive")
    if args.landscape_resolution < 3 or args.landscape_span <= 0:
        parser.error(
            "--landscape-resolution must be at least 3 and --landscape-span must be positive"
        )
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    output_dir = ensure_dir(config["output_dir"])
    mode = config.get("mode", "restoration")
    model_cfg = config["model"]
    train_cfg = config["training"]
    loss_cfg = config["loss"]

    model = ResidualDecorrupter(
        base_channels=int(model_cfg.get("base_channels", 24)),
        residual_scale=float(model_cfg.get("residual_scale", 0.25)),
    ).to(device)
    if model_cfg.get("decorrupter_init"):
        load_model_state(model, model_cfg["decorrupter_init"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
    )
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    if mode == "restoration":
        loader = clean_loader(config, device, True)
        clean_batches = None
        segmenter = None
    elif mode == "task_aware":
        data_cfg = config["data"]
        loader = DataLoader(
            ACDCDataset(data_cfg["root"], data_cfg["split"], data_cfg["image_size"]),
            batch_size=int(data_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(data_cfg.get("workers", 4)),
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
        clean_batches = cycle(clean_loader(config, device, False))
        segmenter = SegFormerSegmenter(
            model_cfg["pretrained_name"],
            int(model_cfg.get("num_classes", 19)),
        ).to(device)
        load_model_state(segmenter, model_cfg["segmenter_checkpoint"])
        segmenter.freeze()
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if (args.preview or args.record_learning or args.record_activations) and mode != "restoration":
        raise ValueError(
            "--preview and --record-learning are supported only for restoration training"
        )
    preview_enabled = args.preview
    recording_enabled = args.record_learning
    activation_enabled = args.record_activations
    if preview_enabled or recording_enabled or activation_enabled:
        preview_item = loader.dataset[0]
        preview_corrupted = preview_item["corrupted"].to(device)
        preview_clean = preview_item["clean"]
        with torch.no_grad():
            preview_restored = model(preview_corrupted.unsqueeze(0)).squeeze(0)
        if preview_enabled:
            preview_enabled = show_reconstruction_preview(
                preview_item["corrupted"],
                preview_restored,
                preview_clean,
                "Before training",
            )
            if not preview_enabled:
                close_reconstruction_preview()
    if recording_enabled:
        reconstruction_video = VideoRecorder(output_dir / "reconstruction_learning.mp4")
        landscape_video = VideoRecorder(output_dir / "loss_landscape_learning.mp4")
        landscape_directions = normalized_directions(model)
        reconstruction_video.write(
            reconstruction_frame(
                preview_item["corrupted"],
                preview_restored,
                preview_clean,
                "Before training",
            )
        )
        initial_landscape = compute_local_loss_landscape(
            model,
            preview_corrupted.unsqueeze(0),
            preview_clean.unsqueeze(0).to(device),
            loss_cfg,
            resolution=args.landscape_resolution,
            span=args.landscape_span,
            directions=landscape_directions,
        )
        landscape_video.write(
            cv2.cvtColor(
                render_loss_landscape(initial_landscape, "Before training"),
                cv2.COLOR_RGB2BGR,
            )
        )
    if activation_enabled:
        activation_capture = ActivationCapture(model)
        activation_video = VideoRecorder(output_dir / "activation_learning.mp4")
        with torch.no_grad():
            preview_restored = model(preview_corrupted.unsqueeze(0)).squeeze(0)
        activation_video.write(
            activation_dashboard_frame(
                preview_item["corrupted"],
                preview_restored,
                activation_capture.activations,
                "Before training",
            )
        )

    best = float("inf")
    step = 0
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"{mode} {epoch}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                if mode == "restoration":
                    corrupted = batch["corrupted"].to(device, non_blocking=True)
                    clean = batch["clean"].to(device, non_blocking=True)
                    loss, _ = reconstruction_loss(
                        model(corrupted),
                        clean,
                        l1_weight=float(loss_cfg.get("l1", 1.0)),
                        ssim_weight=float(loss_cfg.get("ssim", 0.25)),
                        smoothness_weight=float(loss_cfg.get("smoothness", 0.01)),
                    )
                else:
                    adverse = batch["image"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True)
                    clean = next(clean_batches)["image"].to(device, non_blocking=True)
                    restored = model(adverse)
                    loss, _ = task_aware_loss(
                        segmenter(restored),
                        masks,
                        restored,
                        adverse,
                        clean_identity=clean,
                        clean_restored=model(clean),
                        ignore_index=int(config["data"].get("ignore_index", 255)),
                        segmentation_weight=float(loss_cfg.get("segmentation", 1.0)),
                        identity_weight=float(loss_cfg.get("identity", 0.1)),
                        smoothness_weight=float(loss_cfg.get("smoothness", 0.01)),
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(train_cfg.get("grad_clip", 1.0)),
            )
            scaler.step(optimizer)
            scaler.update()
            step += 1
            running += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if preview_enabled and step % args.preview_every == 0:
                model.eval()
                with torch.no_grad():
                    preview_restored = model(preview_corrupted.unsqueeze(0)).squeeze(0)
                model.train()
                preview_enabled = show_reconstruction_preview(
                    preview_item["corrupted"],
                    preview_restored,
                    preview_clean,
                    f"Epoch {epoch} | step {step} | loss {loss.item():.4f}",
                )
                if not preview_enabled:
                    close_reconstruction_preview()
            if recording_enabled and (
                step % args.record_every == 0 or step % args.landscape_every == 0
            ):
                model.eval()
                with torch.no_grad():
                    preview_restored = model(preview_corrupted.unsqueeze(0)).squeeze(0)
                if step % args.record_every == 0:
                    reconstruction_video.write(
                        reconstruction_frame(
                            preview_item["corrupted"],
                            preview_restored,
                            preview_clean,
                            f"Epoch {epoch} | step {step} | loss {loss.item():.4f}",
                        )
                    )
                if step % args.landscape_every == 0:
                    landscape = compute_local_loss_landscape(
                        model,
                        preview_corrupted.unsqueeze(0),
                        preview_clean.unsqueeze(0).to(device),
                        loss_cfg,
                        resolution=args.landscape_resolution,
                        span=args.landscape_span,
                        directions=landscape_directions,
                    )
                    landscape_video.write(
                        cv2.cvtColor(
                            render_loss_landscape(landscape, f"Epoch {epoch} | step {step}"),
                            cv2.COLOR_RGB2BGR,
                        )
                    )
                model.train()
            if activation_enabled and step % args.activation_every == 0:
                model.eval()
                with torch.no_grad():
                    preview_restored = model(preview_corrupted.unsqueeze(0)).squeeze(0)
                activation_video.write(
                    activation_dashboard_frame(
                        preview_item["corrupted"],
                        preview_restored,
                        activation_capture.activations,
                        f"Epoch {epoch} | step {step} | loss {loss.item():.4f}",
                    )
                )
                model.train()

        mean_loss = running / max(1, len(loader))
        metrics = {"loss": mean_loss}
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )
        if mean_loss < best:
            best = mean_loss
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
    if args.preview:
        close_reconstruction_preview()
    if recording_enabled:
        model.eval()
        with torch.no_grad():
            preview_restored = model(preview_corrupted.unsqueeze(0)).squeeze(0)
        final_frame = reconstruction_frame(
            preview_item["corrupted"],
            preview_restored,
            preview_clean,
            "Post-training reconstruction",
        )
        reconstruction_video.write(final_frame)
        cv2.imwrite(str(output_dir / "post_training_reconstruction.png"), final_frame)
        final_landscape = compute_local_loss_landscape(
            model,
            preview_corrupted.unsqueeze(0),
            preview_clean.unsqueeze(0).to(device),
            loss_cfg,
            resolution=args.landscape_resolution,
            span=args.landscape_span,
            directions=landscape_directions,
        )
        landscape_video.write(
            cv2.cvtColor(
                render_loss_landscape(final_landscape, "Post-training"),
                cv2.COLOR_RGB2BGR,
            )
        )
        save_loss_landscape(
            final_landscape,
            "Post-training local loss landscape",
            output_dir / "post_training_loss_landscape.png",
        )
        reconstruction_video.close()
        landscape_video.close()
    if activation_enabled:
        model.eval()
        with torch.no_grad():
            preview_restored = model(preview_corrupted.unsqueeze(0)).squeeze(0)
        activation_video.write(
            activation_dashboard_frame(
                preview_item["corrupted"],
                preview_restored,
                activation_capture.activations,
                "Post-training activations",
            )
        )
        activation_video.close()
        activation_capture.close()


if __name__ == "__main__":
    main()
