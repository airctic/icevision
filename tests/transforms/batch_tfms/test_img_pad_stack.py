import pytest
from mantisshrimp.imports import *
from mantisshrimp import *


@pytest.fixture()
def records():
    imgs = [np.ones((2, 4, 3), dtype=np.uint8), np.ones((3, 2, 3), dtype=np.uint8)]
    return [{"img": img} for img in imgs]


def test_img_pad_stack(records):
    tfmed_records = batch_tfms.ImgPadStack()(records)

    imgs = np.asarray([record["img"] for record in tfmed_records])
    expected = np.zeros((2, 3, 4, 3))
    expected[0, :2, :4, :3] = 1
    expected[1, :3, :2, :3] = 1
    np.testing.assert_equal(imgs, expected)

    before_shapes = [record["shape_before_img_pad_stack"] for record in tfmed_records]
    assert before_shapes == [(2, 4, 3), (3, 2, 3)]
