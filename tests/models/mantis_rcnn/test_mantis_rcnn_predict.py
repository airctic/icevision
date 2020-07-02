import pytest
from mantisshrimp.imports import *
from mantisshrimp import *


@pytest.fixture
def sample_images(samples_source):
    images_dir = samples_source / "images"
    images_files = get_image_files(images_dir)[-2:]
    return [open_img(path) for path in images_files]


@pytest.fixture()
def pretrained_state_dict():
    state_dict = torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
        progress=True,
    )
    return state_dict


def test_mantis_mask_rcnn_predict(sample_images, pretrained_state_dict):
    model = MantisMaskRCNN(91, min_size=128, max_size=128)
    model.load_state_dict(pretrained_state_dict)

    preds = model.predict(sample_images)

    assert len(preds) == 2

    pred = preds[0]
    assert isinstance(pred["labels"], np.ndarray)
    assert isinstance(pred["bboxes"][0], BBox)
    assert isinstance(pred["masks"][0], MaskArray)
    assert isinstance(pred["scores"], np.ndarray)
