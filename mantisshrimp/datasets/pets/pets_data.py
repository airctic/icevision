__all__ = ["CATEGORIES", "load"]

from mantisshrimp.imports import *
from mantisshrimp.utils import *

# CATEGORIES = {
#     "Abyssinian",
#     "great_pyrenees",
#     "Bombay",
#     "Persian",
#     "samoyed",
#     "Maine_Coon",
#     "havanese",
#     "beagle",
#     "yorkshire_terrier",
#     "pomeranian",
#     "scottish_terrier",
#     "saint_bernard",
#     "Siamese",
#     "chihuahua",
#     "Birman",
#     "american_pit_bull_terrier",
#     "miniature_pinscher",
#     "japanese_chin",
#     "British_Shorthair",
#     "Bengal",
#     "Russian_Blue",
#     "newfoundland",
#     "wheaten_terrier",
#     "Ragdoll",
#     "leonberger",
#     "english_cocker_spaniel",
#     "english_setter",
#     "staffordshire_bull_terrier",
#     "german_shorthaired",
#     "Egyptian_Mau",
#     "boxer",
#     "shiba_inu",
#     "keeshond",
#     "pug",
#     "american_bulldog",
#     "basset_hound",
#     "Sphynx",
# }
CATEGORIES = sorted({"dog", "cat"})


def load():
    save_dir = get_data_dir() / "pets"
    save_dir.mkdir(exist_ok=True)

    url_images = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    images_tar_file = save_dir / "pets.tar.gz"
    if not images_tar_file.exists():
        download_url(url=url_images, save_path=images_tar_file)
        shutil.unpack_archive(images_tar_file, save_dir)

    url_annotations = (
        "http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
    )
    annotations_tar_file = save_dir / "annotations.tar.gz"
    if not annotations_tar_file.exists():
        download_url(url=url_annotations, save_path=annotations_tar_file)
        shutil.unpack_archive(annotations_tar_file, save_dir)

    return save_dir
