import torch

from task_aware_decorruption.classical import ClassicalEnhancer


def test_classical_enhancer() -> None:
    images = torch.rand(2, 3, 32, 48)
    output = ClassicalEnhancer()(images)
    assert output.shape == images.shape
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
