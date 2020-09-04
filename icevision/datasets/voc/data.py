__all__ = ["class_map", "load"]

from icevision.imports import *
from icevision.core import *
from icevision.utils import *

_CLASSES = sorted(
    {
        "person",
        "bird",
        "cat",
        "cow",
        "dog",
        "horse",
        "sheep",
        "aeroplane",
        "bicycle",
        "boat",
        "bus",
        "car",
        "motorbike",
        "train",
        "bottle",
        "chair",
        "diningtable",
        "pottedplant",
        "sofa",
        "tvmonitor",
    }
)


def class_map(background: int = 0):
    return ClassMap(classes=_CLASSES, background=background)


def load(force_download=False):
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

    save_dir = get_data_dir() / "voc"
    save_dir.mkdir(exist_ok=True)
    tar_file = save_dir / "trainval.tar"

    if not tar_file.exists() or force_download:
        download_url(url=url, save_path=tar_file)
        # extract file
        shutil.unpack_archive(str(tar_file), str(save_dir))
        # move extract files so they are placed at save_dir
        files_dir = save_dir / "VOCdevkit/VOC2012"
        for file in files_dir.ls():
            shutil.move(str(file), str(save_dir))
        shutil.rmtree(str(files_dir.parent))

    return save_dir
