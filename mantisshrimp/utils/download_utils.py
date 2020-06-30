__all__ = ["download_url", "download_and_extract"]

from mantisshrimp.imports import *
import requests


def download_url(url, save_path, chunk_size=1024) -> None:
    """ Download file from url
    """
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        bar = tqdm(unit="B", total=int(r.headers["Content-Length"]))
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                bar.update(len(chunk))
                f.write(chunk)


def download_and_extract(
    url: Union[str, Path], save_path: Union[str, Path], chunk_size: int = 1024
) -> None:
    save_dir = Path(save_path).parent

    download_url(url=str(url), save_path=str(save_path), chunk_size=chunk_size)
    shutil.unpack_archive(filename=str(save_path), extract_dir=str(save_dir))
