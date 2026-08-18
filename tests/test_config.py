from pathlib import Path

from task_aware_decorruption.config import load_config


def test_load_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 42\n", encoding="utf-8")
    assert load_config(path) == {"seed": 42}
