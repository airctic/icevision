__all__ = ["COCOImageInfoParser"]

from mantisshrimp.imports import *
from mantisshrimp.parsers.defaults import *


class COCOImageInfoParser(ImageInfoParser):
    def __init__(self, infos, img_dir):
        super().__init__()
        self.infos = infos
        self.img_dir = img_dir

    def __iter__(self):
        yield from self.infos

    def __len__(self):
        return len(self.infos)

    def imageid(self, o) -> int:
        return o["id"]

    def filepath(self, o) -> Union[str, Path]:
        return self.img_dir / o["file_name"]

    def height(self, o) -> int:
        return o["height"]

    def width(self, o) -> int:
        return o["width"]
