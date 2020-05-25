import pytest
from mantisshrimp import *
from mantisshrimp.imports import torch, tensor
from mantisshrimp.core.item import _fake_box

@pytest.fixture
def item():
    parser = test_utils.sample_data_parser()
    with np_local_seed(42): train_rs,valid_rs = parser.parse(show_pbar=False)
    return Item.from_record(train_rs[0])

def test_item(item):
    assert item.imageid == 0
    assert item.img.shape == (427,640,3)
    assert item.masks.shape == (16,427,640)
    assert item.labels == [6, 1, 1, 1, 1, 1, 1, 1, 27, 27, 1, 3, 27, 1, 27, 27]
    assert item.iscrowds == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert item.bboxes[0].xyxy == [0, 73.89, 416.44, 379.02]

def test_item2tensor(item):
    x,y = item2tensor(item)
    assert x.dtype == torch.float32
    assert x.shape == (3,427,640)
    assert isinstance(y, dict)
    assert list(y.keys()) == ['image_id', 'labels', 'iscrowd', 'boxes', 'area', 'masks']
    assert y['image_id'].dtype == torch.int64
    assert y['labels'].dtype == torch.int64
    assert y['labels'].shape == (16,)
    assert (y['labels'] == tensor([ 6,  1,  1,  1,  1,  1,  1,  1, 27, 27,  1,  3, 27,  1, 27, 27])).all()
    assert y['iscrowd'].dtype == torch.uint8
    assert y['iscrowd'].shape == (16,)
    assert (y['iscrowd'] == tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()
    assert y['boxes'].dtype == torch.float32
    assert y['boxes'].shape == (16,4)
    assert y['area'].shape == (16,)
    assert y['masks'].dtype == torch.uint8
    assert y['masks'].shape == (16,427,640)


def test_item2tensor_empty(item):
    item = item.replace(bboxes=[], labels=[], iscrowds=[])
    x, y = item2tensor(item)
    assert (y['boxes'] == tensor([_fake_box], dtype=torch.float)).all()
    assert y['labels'] == tensor([0])
    assert y['iscrowd'] == tensor([0])
    assert y['area'] == tensor([4])

