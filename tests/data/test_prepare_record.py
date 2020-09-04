import pytest
from icevision.all import *


@pytest.fixture()
def filepath(samples_source):
    return samples_source / "voc/JPEGImages/2007_000063.jpg"


def test_prepare_img(filepath):
    record = {"filepath": filepath}
    record = default_prepare_record(record)

    assert set(record.keys()) == {"filepath", "img", "height", "width"}
    assert record["img"].shape == (375, 500, 3)
    assert (record["width"], record["height"]) == (500, 375)
