import pytest, requests, PIL
from icevision import *
from icevision.imports import *
from icevision.models.rcnn import faster_rcnn
from icevision.models import efficientdet
import albumentations as A


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


@pytest.fixture(scope="module")
def sample(record):
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


@pytest.fixture()
def fridge_efficientdet_records(samples_source):
    IMG_SIZE = 512
    filepath = samples_source / "fridge/odFridgeObjects/images/10.jpg"

    img = open_img(filepath)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = normalize_imagenet(img)

    labels = [2, 3]
    bboxes = [BBox.from_xyxy(88, 78, 221, 337), BBox.from_xyxy(153, 289, 456, 376)]

    record = {
        "filepath": filepath,
        "imageid": 10,
        "img": img,
        "height": IMG_SIZE,
        "width": IMG_SIZE,
        "labels": labels,
        "bboxes": bboxes,
    }

    return [record]


@pytest.fixture()
def fridge_efficientdet_model() -> nn.Module:
    WEIGHTS_URL = "https://github.com/airctic/model_zoo/releases/download/fridge_tf_efficientdet_lite0/fridge_tf_efficientdet_lite0.zip"
    model = efficientdet.model("tf_efficientdet_lite0", num_classes=5, img_size=512)

    state_dict = torch.hub.load_state_dict_from_url(
        WEIGHTS_URL, map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)

    return model


@pytest.fixture()
def fridge_faster_rcnn_model() -> nn.Module:
    backbone = faster_rcnn.backbones.resnet_fpn.resnet18(pretrained=False)
    return faster_rcnn.model(num_classes=5, backbone=backbone)


@pytest.fixture(scope="session")
def fridge_ds() -> Tuple[Dataset, Dataset]:
    IMG_SIZE = 512
    class_map = datasets.fridge.class_map()
    data_dir = datasets.fridge.load()
    parser = datasets.fridge.parser(data_dir, class_map)

    data_splitter = RandomSplitter([0.8, 0.2])
    train_records, valid_records = parser.parse(data_splitter)

    tfms_ = tfms.A.Adapter([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize()])

    train_ds = Dataset(train_records[:4], tfms_)
    valid_ds = Dataset(valid_records[:4], tfms_)

    return train_ds, valid_ds


@pytest.fixture()
def fridge_efficientdet_dls(fridge_ds) -> Tuple[DataLoader, DataLoader]:
    train_ds, valid_ds = fridge_ds
    train_dl = efficientdet.train_dl(train_ds, batch_size=2)
    valid_dl = efficientdet.valid_dl(valid_ds, batch_size=2)

    return train_dl, valid_dl


@pytest.fixture()
def fridge_faster_rcnn_dls(fridge_ds) -> Tuple[DataLoader, DataLoader]:
    train_ds, valid_ds = fridge_ds
    train_dl = faster_rcnn.train_dl(train_ds, batch_size=2)
    valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=2)

    return train_dl, valid_dl
