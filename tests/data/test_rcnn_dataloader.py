from mantisshrimp.imports import first, tensor, Tensor
from mantisshrimp import *


def test_rcnn_dataloader():
    train_dl, valid_dl = test_utils.sample_rcnn_dataloaders()
    xb,yb = first(train_dl)
    assert len(xb) == 2
    assert len(yb) == 2

    x = xb[0]
    assert x.shape == (3,427,640)
    assert isinstance(x, Tensor)

    y = yb[0]
    assert y['image_id'] == tensor(0)
    assert allequal([isinstance(o, Tensor) for o in y.values()])
    assert (y['labels'] == tensor([6,  1,  1,  1,  1,  1,  1,  1, 27, 27,  1,  3, 27,  1, 27, 27])).all()
    assert (y['iscrowd'] == tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()
    assert (y['boxes'][0] == tensor([0.0000,  73.8900, 416.4400, 379.0200])).all()
    assert y['masks'].shape == (16,427,640)
    assert not (y['masks']==0).all()
    assert allequal(lmap(len, [y['labels'],y['iscrowd'],y['boxes'],y['masks']]))
