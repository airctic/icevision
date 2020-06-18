__all__ = ["download_url"]

from mantisshrimp.imports import *
import requests


def download_url(url, save_path, chunk_size=1024):
    """ Download file from url
    """
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        bar = tqdm(unit="B", total=int(r.headers["Content-Length"]))
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:  # filter out keep-alive new chunks
                bar.update(len(chunk))
                f.write(chunk)
    return save_path
