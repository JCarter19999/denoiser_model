from collections import defaultdict
from collections.abc import Iterable, Callable

import torch
from tqdm import tqdm

from .metrics import SegmentationMetrics


@torch.no_grad()
def evaluate_segmentation(
    segmenter: torch.nn.Module,
    loader: Iterable[dict[str, object]],
    device: torch.device,
    *,
    num_classes: int,
    ignore_index: int,
    preprocessor: Callable[[torch.Tensor], torch.Tensor] | None = None,
    description: str = "evaluate",
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    segmenter.eval()
    overall = SegmentationMetrics(num_classes, ignore_index)
    by_condition: dict[str, SegmentationMetrics] = defaultdict(
        lambda: SegmentationMetrics(num_classes, ignore_index)
    )
    for batch in tqdm(loader, desc=description):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        if preprocessor is not None:
            images = preprocessor(images)
        logits = segmenter(images)
        overall.update(logits, masks)
        predictions = logits.argmax(dim=1)
        for index, condition in enumerate(batch["condition"]):
            by_condition[str(condition)].update(predictions[index : index + 1], masks[index : index + 1])
    return overall.compute(), {name: metric.compute() for name, metric in by_condition.items()}
