import pytest
from icevision.all import *


# WARNING: Only works with cuda: https://github.com/rwightman/efficientdet-pytorch/issues/44#issuecomment-662594014
@pytest.mark.cuda
@pytest.mark.parametrize("metrics", [[COCOMetric()]])
def test_fastai_efficientdet_train(
    fridge_efficientdet_dls, fridge_efficientdet_model, metrics
):
    learn = efficientdet.fastai.learner(
        dls=fridge_efficientdet_dls, model=fridge_efficientdet_model, metrics=metrics
    )

    learn.fine_tune(1, 1e-4)
