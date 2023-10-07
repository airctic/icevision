__all__ = ["download_mmseg_configs"]

from icevision.imports import *
from icevision.utils import *

VERSION = "0.29.1"
BASE_URL = f"https://github.com/potipot/icevision/releases/download/0.13.0/mmsegmentation_configs-{VERSION}.zip"


def download_mmseg_configs() -> Path:

    save_dir = get_root_dir() / f"mmsegmentation_configs"

    mmseg_config_path = save_dir / Path(BASE_URL).stem / "configs"
    download_path = save_dir / f"{Path(BASE_URL).stem}.zip"

    if mmseg_config_path.exists():
        logger.info(
            f"The mmseg config folder already exists. No need to download it. Path : {mmseg_config_path}"
        )
    elif download_path.exists():
        # The zip file was downloaded by not extracted yet
        # Extract zip file
        logger.info(f"Extracting the {download_path.name} file.")
        save_dir = Path(download_path).parent
        shutil.unpack_archive(filename=str(download_path), extract_dir=str(save_dir))
    else:
        save_dir.mkdir(parents=True, exist_ok=True)

        if not download_path.exists():
            logger.info("Downloading mmseg configs")
            download_and_extract(BASE_URL, download_path)

    return mmseg_config_path
