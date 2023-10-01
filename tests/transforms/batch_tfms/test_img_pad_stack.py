import pytest
from icevision.all import *


@pytest.fixture()
def records():
    imgs = [np.ones((2, 4, 3), dtype=np.float32), np.ones((3, 2, 3), dtype=np.float32)]

    records = []
    for img in imgs:
        record = BaseRecord((ImageRecordComponent(),))
        record.set_img(img)
        records.append(record)

    return records


@pytest.mark.parametrize("pad_value", [0, (1, 2, 3)])
def test_img_pad_stack(records, pad_value):
    tfmed_records = tfms.batch.ImgPadStack(pad_value=pad_value)(records)
    imgs = np.asarray([record.img for record in tfmed_records])

    expected = np.ones((2, 3, 4, 3), dtype=np.float32)
    expected *= np.array(pad_value).reshape(-1)
    expected[0, :2, :4, :3] = 1
    expected[1, :3, :2, :3] = 1
    np.testing.assert_equal(imgs, expected)

    before_shapes = [(record.height, record.width) for record in tfmed_records]
    assert before_shapes == [(2, 4), (3, 2)]
