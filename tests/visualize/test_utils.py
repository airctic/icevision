import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "color_input",
    [
        ("red"),
        ("#fff"),
        ("#ffaaaa"),
        ([255, 255, 255]),
        ((255, 255, 255)),
    ],
)
def test_as_rgb_tuple(color_input):
    assert as_rgb_tuple(color_input)
