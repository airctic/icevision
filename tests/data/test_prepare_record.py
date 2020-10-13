import pytest
from icevision.all import *


@pytest.fixture()
def filepath(samples_source):
    return samples_source / "voc/JPEGImages/2007_000063.jpg"


def test_prepare_img(filepath):
    Record = create_mixed_record((FilepathRecordMixin,))
    record = Record()
    record.set_imageid(1)
    record.set_filepath(filepath)
    record = record.load()

    assert set(record.keys()) == {"imageid", "filepath", "img", "height", "width"}
    assert record["img"].shape == (375, 500, 3)
    assert (record["width"], record["height"]) == (500, 375)
