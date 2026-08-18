import torch

from task_aware_decorruption.models.decorrupter import ResidualDecorrupter


def test_decorrupter_shape_range_and_gradients() -> None:
    model = ResidualDecorrupter(base_channels=8, residual_scale=0.2)
    image = torch.rand(2, 3, 31, 47, requires_grad=True)
    restored = model(image)
    assert restored.shape == image.shape
    assert float(restored.detach().min()) >= 0.0
    assert float(restored.detach().max()) <= 1.0
    restored.mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
