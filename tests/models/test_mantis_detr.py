import pytest
import albumentations as A
import requests
from mantisshrimp.imports import np, im2tensor
from mantisshrimp import *
from PIL import Image


@pytest.mark.slow
def test_detr_result():
    model = MantisDetr().eval()
    tfm = AlbuTransform([A.SmallestMaxSize(800), A.Normalize(*imagenet_stats)])
    # get image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = np.array(Image.open(requests.get(url, stream=True).raw))
    tfmed_img = tfm.apply(img)["img"]
    tensor_img = im2tensor(tfmed_img)
    # get predictions
    probs, bboxes = model.predict(tensor_img)

    assert len(bboxes) == len(probs) == 4
    assert probs.argmax(-1).tolist() == [17, 75, 17, 75]
    assert bboxes[0].xyxy == [24, 87, 517, 791]
