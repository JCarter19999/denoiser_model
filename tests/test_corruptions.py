import random

import torch

from task_aware_decorruption.data.corruptions import CorruptionPipeline


def test_corruptions_preserve_shape_and_range() -> None:
    random.seed(4)
    torch.manual_seed(4)
    image = torch.rand(3, 32, 48)
    pipeline = CorruptionPipeline((0.5, 0.5))
    for _ in range(14):
        result, name, severity = pipeline(image)
        assert result.shape == image.shape
        assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0
        assert name in pipeline.names
        assert severity == 0.5
