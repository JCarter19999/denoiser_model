from pathlib import Path

from PIL import Image

from task_aware_decorruption.data.clean import CleanImageDataset
from task_aware_decorruption.data.corruptions import CorruptionPipeline


def test_clean_dataset(tmp_path: Path) -> None:
    Image.new("RGB", (24, 12), (100, 110, 120)).save(tmp_path / "image.png")
    dataset = CleanImageDataset(tmp_path, (8, 16), CorruptionPipeline((0.4, 0.4)))
    item = dataset[0]
    assert item["clean"].shape == (3, 8, 16)
    assert item["corrupted"].shape == (3, 8, 16)


def test_clean_dataset_filters_by_filename_pattern(tmp_path: Path) -> None:
    reference_dir = tmp_path / "fog" / "train_ref"
    reference_dir.mkdir(parents=True)
    Image.new("RGB", (24, 12)).save(reference_dir / "reference_rgb_ref_anon.png")
    Image.new("RGB", (24, 12)).save(tmp_path / "adverse_rgb_anon.png")

    dataset = CleanImageDataset(
        tmp_path,
        (8, 16),
        filename_pattern="*/train_ref/*_rgb_ref_anon.png",
    )

    assert len(dataset) == 1
    assert dataset.paths[0].name == "reference_rgb_ref_anon.png"
