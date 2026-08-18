from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


CONDITIONS = ("fog", "night", "rain", "snow")


@dataclass(frozen=True)
class ACDCSample:
    image_path: Path
    mask_path: Path
    condition: str


def _first_with_matching_files(paths: list[Path], pattern: str) -> Path | None:
    for path in paths:
        if path.is_dir() and any(path.rglob(pattern)):
            return path
    return None


def discover_acdc_pairs(
    root: str | Path,
    split: str,
) -> list[ACDCSample]:
    root_path = Path(root)
    samples: list[ACDCSample] = []

    for condition in CONDITIONS:
        image_root = _first_with_matching_files(
            [
                root_path / "rgb_anon" / split / condition,
                root_path / "rgb_anon" / condition / split,
            ],
            "*_rgb_anon.png",
        )

        mask_root = _first_with_matching_files(
            [
                root_path / "gt" / split / condition,
                root_path / "gt" / condition / split,
            ],
            "*_gt_labelTrainIds.png",
        )

        if image_root is None or mask_root is None:
            continue

        for image_path in sorted(image_root.rglob("*_rgb_anon.png")):
            relative = image_path.relative_to(image_root)
            mask_name = image_path.name.replace(
                "_rgb_anon.png",
                "_gt_labelTrainIds.png",
            )
            mask_path = mask_root / relative.parent / mask_name

            if mask_path.is_file():
                samples.append(
                    ACDCSample(
                        image_path=image_path,
                        mask_path=mask_path,
                        condition=condition,
                    )
                )

    if not samples:
        checked = [
            root_path / "rgb_anon" / condition / split
            for condition in CONDITIONS
        ]
        checked.extend(
            root_path / "gt" / split / condition
            for condition in CONDITIONS
        )

        checked_text = "`n".join(str(path) for path in checked)

        raise FileNotFoundError(
            f"No ACDC image-mask pairs found under {root_path} "
            f"for split '{split}'. Checked:`n{checked_text}"
        )

    return sorted(samples, key=lambda sample: str(sample.image_path))


class ACDCDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: list[int] | tuple[int, int],
    ) -> None:
        self.samples = discover_acdc_pairs(root, split)
        self.size = (int(image_size[0]), int(image_size[1]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]

        image = Image.open(sample.image_path).convert("RGB")
        mask = Image.open(sample.mask_path)

        image_tensor = TF.to_tensor(
            TF.resize(
                image,
                self.size,
                antialias=True,
            )
        )

        mask_tensor = TF.pil_to_tensor(
            TF.resize(
                mask,
                self.size,
                interpolation=InterpolationMode.NEAREST,
            )
        ).squeeze(0).long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "condition": sample.condition,
            "path": str(sample.image_path),
        }
