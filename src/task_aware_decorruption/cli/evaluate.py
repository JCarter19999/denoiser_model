import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from task_aware_decorruption.classical import ClassicalEnhancer
from task_aware_decorruption.config import load_config
from task_aware_decorruption.data import ACDCDataset
from task_aware_decorruption.engine import evaluate_segmentation
from task_aware_decorruption.models import ResidualDecorrupter, SegFormerSegmenter
from task_aware_decorruption.utils import ensure_dir, load_model_state, resolve_device, seed_everything, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("none", "classical", "learned"), required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    data_cfg = config["data"]
    model_cfg = config["model"]
    loader = DataLoader(
        ACDCDataset(data_cfg["root"], data_cfg["split"], data_cfg["image_size"]),
        batch_size=int(data_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(data_cfg.get("workers", 4)),
        pin_memory=device.type == "cuda",
    )
    segmenter = SegFormerSegmenter(model_cfg["pretrained_name"], int(model_cfg.get("num_classes", 19))).to(device)
    load_model_state(segmenter, model_cfg["segmenter_checkpoint"])
    segmenter.eval()

    preprocessor = None
    if args.method == "classical":
        preprocessor = ClassicalEnhancer(**config.get("classical", {}))
    elif args.method == "learned":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for learned preprocessing")
        preprocessor = ResidualDecorrupter(
            base_channels=int(model_cfg.get("base_channels", 24)),
            residual_scale=float(model_cfg.get("residual_scale", 0.25)),
        ).to(device)
        load_model_state(preprocessor, args.checkpoint)
        preprocessor.eval()

    overall, conditions = evaluate_segmentation(
        segmenter,
        loader,
        device,
        num_classes=int(model_cfg.get("num_classes", 19)),
        ignore_index=int(data_cfg.get("ignore_index", 255)),
        preprocessor=preprocessor,
        description=args.method,
    )
    output_dir = ensure_dir(config["output_dir"])
    name = args.method
    if args.checkpoint:
        name += f"_{Path(args.checkpoint).parent.name}"
    write_json(output_dir / f"{name}.json", {"overall": overall, "conditions": conditions})
    print(f"mIoU={overall['miou']:.4f}")


if __name__ == "__main__":
    main()
