from fnmatch import fnmatch
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .corruptions import CorruptionPipeline


class CleanImageDataset(Dataset[dict[str, object]]):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root: str | Path,
        image_size: list[int] | tuple[int, int],
        corruption: CorruptionPipeline | None = None,
        filename_pattern: str | None = None,
    ) -> None:
        root_path = Path(root)
        image_paths = (
            path for path in root_path.rglob("*") if path.suffix.lower() in self.extensions
        )
        if filename_pattern:
            image_paths = (
                path
                for path in image_paths
                if fnmatch(path.relative_to(root_path).as_posix(), filename_pattern)
            )
        self.paths = sorted(image_paths)
        if not self.paths:
            pattern_text = f" matching {filename_pattern!r}" if filename_pattern else ""
            raise RuntimeError(f"No images found under {root_path}{pattern_text}")
        self.size = (int(image_size[0]), int(image_size[1]))
        self.corruption = corruption

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        clean = TF.to_tensor(TF.resize(Image.open(path).convert("RGB"), self.size, antialias=True))
        if self.corruption is None:
            return {"image": clean, "path": str(path)}
        corrupted, name, severity = self.corruption(clean)
        return {
            "clean": clean,
            "corrupted": corrupted,
            "corruption": name,
            "severity": severity,
            "path": str(path),
        }
