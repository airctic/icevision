import pytest
from icevision.imports import *
from icevision import *
from icevision.models.torchvision_models import mask_rcnn


@pytest.fixture
def sample_dataset(samples_source):
    images_dir = samples_source / "images"
    images_files = get_image_files(images_dir)[-2:]

    images = [open_img(path) for path in images_files]
    images = [cv2.resize(image, (128, 128)) for image in images]

    return Dataset.from_images(images)


@pytest.fixture()
def pretrained_state_dict():
    state_dict = torch.hub.load_state_dict_from_url(
        "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
        progress=True,
    )
    return state_dict


def _test_preds(preds):
    assert len(preds) == 2

    pred = preds[0]
    assert isinstance(pred["labels"], np.ndarray)
    assert isinstance(pred["bboxes"][0], BBox)
    assert isinstance(pred["masks"][0], MaskArray)
    assert isinstance(pred["scores"], np.ndarray)


def test_mantis_mask_rcnn_predict(sample_dataset, pretrained_state_dict):
    model = mask_rcnn.model(num_classes=91)
    model.load_state_dict(pretrained_state_dict)

    batch, samples = mask_rcnn.build_infer_batch(dataset=sample_dataset)
    preds = mask_rcnn.predict(model=model, batch=batch)

    _test_preds(preds)


def test_mantis_mask_rcnn_predict_dl(sample_dataset, pretrained_state_dict):
    model = mask_rcnn.model(num_classes=91)
    model.load_state_dict(pretrained_state_dict)

    infer_dl = mask_rcnn.infer_dl(dataset=sample_dataset, batch_size=2)
    samples, preds = mask_rcnn.predict_dl(
        model=model, infer_dl=infer_dl, show_pbar=False
    )

    _test_preds(preds)


def test_mantis_mask_rcnn_predict_dl_threshold(sample_dataset, pretrained_state_dict):
    model = mask_rcnn.model(num_classes=91)
    model.load_state_dict(pretrained_state_dict)

    infer_dl = mask_rcnn.infer_dl(dataset=sample_dataset, batch_size=2)
    samples, preds = mask_rcnn.predict_dl(
        model=model,
        infer_dl=infer_dl,
        show_pbar=False,
        detection_threshold=1.0,
    )

    assert len(preds[0]["labels"]) == 0
