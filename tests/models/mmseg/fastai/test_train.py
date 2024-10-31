import pytest
from icevision.all import *

from icevision.models.mmseg.models.segformer.backbones import *
from icevision.models.mmseg.models import *


@pytest.mark.parametrize("backbone", [mit_b0, mit_b1, mit_b2])
def test_fastai_train(camvid_ds, backbone):
    train_ds, valid_ds = camvid_ds
    train_dl = segformer.train_dl(train_ds, batch_size=4, num_workers=0, shuffle=False)
    valid_dl = segformer.valid_dl(valid_ds, batch_size=1, num_workers=0, shuffle=False)
    model = segformer.model(num_classes=32, backbone=backbone(pretrained=True))

    learn = segformer.learner(dls=[train_dl, valid_dl], model=model)

    learn.fine_tune(1, 1e-4)
