__all__ = ["download_mmdet_configs"]

from icevision.imports import *
from icevision.utils import *

VERSION = "v2.10.0"
BASE_URL = "https://codeload.github.com/lgvaz/mmdetection_configs/zip/refs/tags"


def download_mmdet_configs() -> Path:
    save_dir = get_root_dir() / f"mmdetection_configs"
    save_dir.mkdir(parents=True, exist_ok=True)

    download_path = save_dir / f"{VERSION}.zip"
    if not download_path.exists():
        logger.info("Downloading mmdet configs")

        download_and_extract(f"{BASE_URL}/{VERSION}", download_path)

    return save_dir / f"mmdetection_configs-{VERSION[1:]}/configs"
