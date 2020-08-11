from mantisshrimp.utils import *


def load():
    images_url = "https://drive.google.com/uc?id=1GDr1OkoXdhaXWGA8S3MAq3a522Tak-nx&export=download"
    lists_url = "https://drive.google.com/uc?export=download&id=1vZuZPqha0JjmwkdaS_XtYryE3Jf5Q1AC"
    annotations_url = "https://drive.google.com/uc?export=download&id=16NsbTpMs5L6hT4hUJAmpW2u7wH326WTR"

    data_dir = get_data_dir() / "birds"
    download_and_extract_gdrive(
        url=images_url, filename="images.tgz",
    )
