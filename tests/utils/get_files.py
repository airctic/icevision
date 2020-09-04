from icevision.all import *


def test_get_image_files():
    fns = get_image_files("../../samples/images")
    assert len(fns) == 6
