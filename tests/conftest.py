"""Shared fixtures.

The real competition data lives in ``data/`` and is used directly (it is only
ever read). Anything the tests *write* goes to ``tmp_path``, so running the
suite never disturbs the submission artifacts.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:  # alongside pytest.ini pythonpath
    sys.path.insert(0, str(SRC))

from common.config import Config, load_config  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def base_config() -> Config:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="session")
def data_available(base_config: Config) -> bool:
    """Skip data-dependent tests when the competition data is absent."""
    return (
        base_config.path("data.instances_json").is_file()
        and base_config.path("data.test_b_manifest").is_file()
    )


@pytest.fixture
def cfg(base_config: Config, tmp_path: Path) -> Config:
    """Config with every *output* path redirected into this test's tmp dir."""
    cfg = Config(copy.deepcopy(dict(base_config)))
    cfg["data"]["processed_dir"] = str(tmp_path / "processed")
    cfg["split"]["output_dir"] = str(tmp_path / "processed" / "splits")
    cfg["eval"]["output_dir"] = str(tmp_path / "reports")
    cfg["submit"]["output_dir"] = str(tmp_path / "submissions")
    cfg["train"]["output_dir"] = str(tmp_path / "checkpoints")
    return cfg


@pytest.fixture(scope="session")
def train_images(base_config: Config):
    from data.coco import load_train_images

    return load_train_images(base_config)


@pytest.fixture(scope="session")
def test_b_images(base_config: Config):
    from submit.result_json import load_test_manifest

    return load_test_manifest(base_config.path("data.test_b_manifest"))
