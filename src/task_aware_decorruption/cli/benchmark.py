import argparse
import time

import torch

from task_aware_decorruption.classical import ClassicalEnhancer
from task_aware_decorruption.config import load_config
from task_aware_decorruption.models import ResidualDecorrupter, SegFormerSegmenter
from task_aware_decorruption.utils import load_model_state, resolve_device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("none", "classical", "learned"), required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()
    config = load_config(args.config)
    device = resolve_device(config.get("device", "auto"))
    data_cfg = config["data"]
    model_cfg = config["model"]
    height, width = map(int, data_cfg["image_size"])
    images = torch.rand(int(data_cfg.get("batch_size", 1)), 3, height, width, device=device)

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

    def run() -> torch.Tensor:
        processed = preprocessor(images) if preprocessor is not None else images
        return segmenter(processed)

    with torch.no_grad():
        for _ in range(args.warmup):
            run()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(args.runs):
            run()
        synchronize(device)
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000 / args.runs
    fps = images.shape[0] * args.runs / elapsed
    parameters = sum(parameter.numel() for parameter in segmenter.parameters())
    if isinstance(preprocessor, torch.nn.Module):
        parameters += sum(parameter.numel() for parameter in preprocessor.parameters())
    print(f"latency_ms={latency_ms:.3f}")
    print(f"fps={fps:.3f}")
    print(f"parameters={parameters}")


if __name__ == "__main__":
    main()
