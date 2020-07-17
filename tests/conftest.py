import pytest, requests, PIL
from mantisshrimp import *
from mantisshrimp.imports import *


@pytest.fixture(scope="module")
def samples_source():
    return Path(__file__).absolute().parent.parent / "samples"


@pytest.fixture(scope="session")
def coco_imageid_map():
    return IDMap()


@pytest.fixture(scope="module")
def records(coco_imageid_map):
    parser = test_utils.sample_combined_parser()
    return parser.parse(idmap=coco_imageid_map)[0]


@pytest.fixture(scope="module")
def record(records):
    return records[0].copy()


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


@pytest.fixture()
def fridge_img(samples_source):
    filepath = samples_source / "fridge/images/10.jpg"
    return open_img(filepath)
