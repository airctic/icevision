import pytest, requests, PIL
from mantisshrimp import *
from mantisshrimp.imports import *


@pytest.fixture(scope="module")
def records():
    parser = test_utils.sample_combined_parser()
    return parser.parse()[0]


@pytest.fixture(scope="module")
def record(records):
    return records[2].copy()


@pytest.fixture(scope="module")
def data_sample(record):
    return default_prepare_record(record)


@pytest.fixture()
def image():
    # Get a big image because of these big CNNs
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = np.array(PIL.Image.open(requests.get(url, stream=True).raw))
    # Get a big size image for these big resnets
    img = cv2.resize(img, (2048, 2048))
    tensor_img = im2tensor(img)
    tensor_img = torch.unsqueeze(tensor_img, 0)
    return tensor_img
