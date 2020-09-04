from icevision.all import *


def test_open_img():
    fn = Path(__file__).parent.parent.parent
    fn = str(fn / "samples/images/000000000089.jpg")
    assert open_img(fn).shape == (480, 640, 3)
    assert open_img(fn, gray=True).shape == (480, 640)
