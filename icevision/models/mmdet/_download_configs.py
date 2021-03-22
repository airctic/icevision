__all__ = ["download_mmdet_configs"]

from icevision.imports import *
from icevision.utils import *


def download_mmdet_configs() -> Path:
    version = "v2.10.0"
    base_url = "https://codeload.github.com/lgvaz/mmdetection_configs/zip/refs/tags"

    save_dir = get_root_dir() / f"mmdetection_configs"
    save_dir.mkdir(parents=True, exist_ok=True)

    download_path = save_dir / f"{version}.zip"
    if not download_path.exists():
        logger.info("Downloading mmdet configs")

        download_and_extract(f"{base_url}/{version}", download_path)

    return save_dir / f"mmdetection_configs-{version}"
