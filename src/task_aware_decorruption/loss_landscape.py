from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .losses import reconstruction_loss


@dataclass(frozen=True)
class LossLandscape:
    coordinates: np.ndarray
    values: np.ndarray


Directions = tuple[list[torch.Tensor], list[torch.Tensor]]


def normalized_directions(model: torch.nn.Module) -> Directions:
    first: list[torch.Tensor] = []
    second: list[torch.Tensor] = []
    for parameter in model.parameters():
        first_direction = torch.randn_like(parameter)
        second_direction = torch.randn_like(parameter)
        parameter_norm = parameter.norm()
        first_norm = first_direction.norm()
        second_norm = second_direction.norm()
        if parameter_norm > 0 and first_norm > 0:
            first_direction = first_direction * (parameter_norm / first_norm)
        else:
            first_direction.zero_()
        if parameter_norm > 0 and second_norm > 0:
            second_direction = second_direction * (parameter_norm / second_norm)
        else:
            second_direction.zero_()
        first.append(first_direction)
        second.append(second_direction)
    return first, second


def compute_local_loss_landscape(
    model: torch.nn.Module,
    corrupted: torch.Tensor,
    clean: torch.Tensor,
    loss_config: dict,
    *,
    resolution: int,
    span: float,
    directions: Directions | None = None,
) -> LossLandscape:
    if resolution < 3:
        raise ValueError("resolution must be at least 3")
    if span <= 0:
        raise ValueError("span must be positive")
    original = [parameter.detach().clone() for parameter in model.parameters()]
    first_direction, second_direction = directions or normalized_directions(model)
    coordinates = np.linspace(-span, span, resolution)
    values = np.empty((resolution, resolution), dtype=np.float32)
    try:
        with torch.no_grad():
            for row, alpha in enumerate(coordinates):
                for column, beta in enumerate(coordinates):
                    for parameter, base, first, second in zip(
                        model.parameters(),
                        original,
                        first_direction,
                        second_direction,
                        strict=True,
                    ):
                        parameter.copy_(base + float(alpha) * first + float(beta) * second)
                    loss, _ = reconstruction_loss(
                        model(corrupted),
                        clean,
                        l1_weight=float(loss_config.get("l1", 1.0)),
                        ssim_weight=float(loss_config.get("ssim", 0.25)),
                        smoothness_weight=float(loss_config.get("smoothness", 0.01)),
                    )
                    values[row, column] = loss.item()
    finally:
        with torch.no_grad():
            for parameter, base in zip(model.parameters(), original, strict=True):
                parameter.copy_(base)
    return LossLandscape(coordinates=coordinates, values=values)


def _local_extrema(values: np.ndarray, find_minimum: bool) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for row in range(1, values.shape[0] - 1):
        for column in range(1, values.shape[1] - 1):
            neighborhood = values[row - 1 : row + 2, column - 1 : column + 2]
            center = values[row, column]
            comparison = neighborhood.min() if find_minimum else neighborhood.max()
            if center == comparison and np.any(neighborhood != center):
                points.append((row, column))
    return points


def render_loss_landscape(landscape: LossLandscape, title: str) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coordinates = landscape.coordinates
    values = landscape.values
    first_grid, second_grid = np.meshgrid(coordinates, coordinates)
    minimum = np.unravel_index(values.argmin(), values.shape)
    maximum = np.unravel_index(values.argmax(), values.shape)
    local_minima = _local_extrema(values, find_minimum=True)
    local_maxima = _local_extrema(values, find_minimum=False)
    figure = plt.figure(figsize=(16, 7))
    surface_axis = figure.add_subplot(1, 2, 1, projection="3d")
    surface = surface_axis.plot_surface(
        first_grid,
        second_grid,
        values,
        cmap="viridis",
        alpha=0.9,
    )
    surface_axis.scatter(
        coordinates[minimum[1]],
        coordinates[minimum[0]],
        values[minimum],
        color="blue",
        s=50,
        label="Global minimum",
    )
    surface_axis.scatter(
        coordinates[maximum[1]],
        coordinates[maximum[0]],
        values[maximum],
        color="red",
        s=50,
        label="Global maximum",
    )
    surface_axis.set(title=title, xlabel="Direction 1", ylabel="Direction 2", zlabel="Loss")
    surface_axis.legend()
    figure.colorbar(surface, ax=surface_axis, shrink=0.6, label="Loss")
    contour_axis = figure.add_subplot(1, 2, 2)
    contour = contour_axis.contourf(first_grid, second_grid, values, levels=20, cmap="viridis")
    for row, column in local_minima:
        contour_axis.scatter(coordinates[column], coordinates[row], color="blue", s=15)
    for row, column in local_maxima:
        contour_axis.scatter(coordinates[column], coordinates[row], color="red", s=15)
    contour_axis.scatter(
        coordinates[minimum[1]],
        coordinates[minimum[0]],
        color="blue",
        s=50,
        label="Minimum",
    )
    contour_axis.scatter(
        coordinates[maximum[1]],
        coordinates[maximum[0]],
        color="red",
        s=50,
        label="Maximum",
    )
    contour_axis.set(
        title="Contour map (blue minima, red maxima)",
        xlabel="Direction 1",
        ylabel="Direction 2",
    )
    contour_axis.legend()
    figure.colorbar(contour, ax=contour_axis, label="Loss")
    figure.tight_layout()
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    return image


def save_loss_landscape(landscape: LossLandscape, title: str, path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.imsave(path, render_loss_landscape(landscape, title))
