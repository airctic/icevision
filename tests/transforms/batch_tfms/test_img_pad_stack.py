import pytest
from mantisshrimp.imports import *
from mantisshrimp import *


@pytest.fixture()
def records():
    imgs = [np.ones((2, 4, 3), dtype=np.float), np.ones((3, 2, 3), dtype=np.float)]
    return [{"img": img} for img in imgs]


@pytest.mark.parametrize("pad_value", [0, (1, 2, 3)])
def test_img_pad_stack(records, pad_value):
    tfmed_records = batch_tfms.ImgPadStack(pad_value=pad_value)(records)
    imgs = np.asarray([record["img"] for record in tfmed_records])

    expected = np.ones((2, 3, 4, 3), dtype=np.float)
    expected *= np.array(pad_value).reshape(-1)
    expected[0, :2, :4, :3] = 1
    expected[1, :3, :2, :3] = 1
    np.testing.assert_equal(imgs, expected)

    before_shapes = [record["shape_before_img_pad_stack"] for record in tfmed_records]
    assert before_shapes == [(2, 4, 3), (3, 2, 3)]
