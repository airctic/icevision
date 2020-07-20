import pytest
from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.models.rcnn import mask_rcnn


@pytest.fixture
def sample_images(samples_source):
    images_dir = samples_source / "images"
    images_files = get_image_files(images_dir)[-2:]

    images = [open_img(path) for path in images_files]
    images = [cv2.resize(image, (128, 128)) for image in images]

    return images


@pytest.fixture()
def pretrained_state_dict():
    state_dict = torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
        progress=True,
    )
    return state_dict


def test_mantis_mask_rcnn_predict(sample_images, pretrained_state_dict):
    model = mask_rcnn.model(num_classes=91)
    model.load_state_dict(pretrained_state_dict)

    batch = mask_rcnn.build_infer_batch(images=sample_images)
    preds = mask_rcnn.predict(model=model, batch=batch)

    assert len(preds) == 2

    pred = preds[0]
    assert isinstance(pred["labels"], np.ndarray)
    assert isinstance(pred["bboxes"][0], BBox)
    assert isinstance(pred["masks"][0], MaskArray)
    assert isinstance(pred["scores"], np.ndarray)
