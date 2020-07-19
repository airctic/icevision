from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.models import efficientdet


def test_efficient_det_predict(fridge_img):
    img = cv2.resize(fridge_img, (512, 512))
    img = normalize_imagenet(img)
    class_map = datasets.fridge.class_map()

    model = efficientdet.model(
        "tf_efficientdet_lite0", num_classes=len(class_map), img_size=512
    )
    weights_url = "https://mantisshrimp-models.s3.us-east-2.amazonaws.com/fridge_tf_efficientdet_lite0.zip"
    state_dict = torch.hub.load_state_dict_from_url(
        weights_url, map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)

    records = [{"img": img, "height": 512, "width": 512}]
    batch = efficientdet.build_infer_batch(records)
    preds = efficientdet.predict(model=model, batch=batch)

    assert len(preds) == 1

    pred = preds[0]
    assert len(pred["scores"]) == 2

    assert pred["labels"] == [2, 3]

    assert len(pred["bboxes"]) == 2
    assert isinstance(pred["bboxes"][0], BBox)
