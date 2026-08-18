from pathlib import Path

from PIL import Image

from task_aware_decorruption.data.acdc import ACDCDataset, discover_acdc_pairs


def test_acdc_discovery_and_load(tmp_path: Path) -> None:
    image_dir = tmp_path / "rgb_anon" / "train" / "fog" / "seq"
    mask_dir = tmp_path / "gt" / "train" / "fog" / "seq"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image_path = image_dir / "frame_rgb_anon.png"
    mask_path = mask_dir / "frame_gt_labelTrainIds.png"
    Image.new("RGB", (20, 10), (10, 20, 30)).save(image_path)
    Image.new("L", (20, 10), 2).save(mask_path)
    samples = discover_acdc_pairs(tmp_path, "train")
    assert len(samples) == 1
    assert samples[0].condition == "fog"
    item = ACDCDataset(tmp_path, "train", (8, 16))[0]
    assert item["image"].shape == (3, 8, 16)
    assert item["mask"].shape == (8, 16)


def test_acdc_discovery_ignores_empty_alternate_layout(tmp_path: Path) -> None:
    (tmp_path / "rgb_anon" / "fog" / "train").mkdir(parents=True)
    image_dir = tmp_path / "rgb_anon" / "train" / "fog" / "seq"
    mask_dir = tmp_path / "gt" / "train" / "fog" / "seq"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (20, 10)).save(image_dir / "frame_rgb_anon.png")
    Image.new("L", (20, 10)).save(mask_dir / "frame_gt_labelTrainIds.png")

    samples = discover_acdc_pairs(tmp_path, "train")

    assert len(samples) == 1
    assert samples[0].image_path.parent == image_dir
