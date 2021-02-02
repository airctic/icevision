import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("voc/JPEGImages/2007_000063.jpg", (375, 500, 3)),
        ("images2/flies.jpeg", (2592, 3888, 3)),
    ],
)
def test_open_img(samples_source, fn, expected):
    assert open_img(samples_source / fn).shape == expected
    assert open_img(samples_source / fn, gray=True).shape == expected[:-1]


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("voc/JPEGImages/2007_000063.jpg", (500, 375)),
        ("images2/flies.jpeg", (3888, 2592)),
    ],
)
def test_get_image_size(samples_source, fn, expected):
    size = get_image_size(samples_source / fn)
    assert size == (expected)
