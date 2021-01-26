import pytest
from icevision import *
from icevision.imports import *
from icevision.models.torchvision import faster_rcnn
from icevision.models.ross import efficientdet
import albumentations as A


@pytest.fixture(scope="session")
def samples_source():
    return Path(__file__).absolute().parent.parent / "samples"


@pytest.fixture(scope="session")
def coco_imageid_map():
    return IDMap()


@pytest.fixture()
def fridge_efficientdet_records(samples_source):
    IMG_SIZE = 384
    filepath = samples_source / "fridge/odFridgeObjects/images/10.jpg"

    img = open_img(filepath)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = normalize_imagenet(img)

    labels = [2, 3]
    bboxes = [BBox.from_xyxy(66, 58, 165, 252), BBox.from_xyxy(114, 216, 342, 282)]

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
    WEIGHTS_URL = "https://github.com/airctic/model_zoo/releases/download/m2/fridge_tf_efficientdet_lite0.zip"
    model = efficientdet.model("tf_efficientdet_lite0", num_classes=5, img_size=384)

    state_dict = torch.hub.load_state_dict_from_url(
        WEIGHTS_URL, map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)

    return model


@pytest.fixture()
def fridge_faster_rcnn_model() -> nn.Module:
    backbone = faster_rcnn.backbones.resnet_fpn.resnet18(pretrained=False)
    return faster_rcnn.model(num_classes=5, backbone=backbone)


@pytest.fixture(scope="module")
def fridge_ds(samples_source, fridge_class_map) -> Tuple[Dataset, Dataset]:
    IMG_SIZE = 384

    parser = parsers.VocXmlParser(
        annotations_dir=samples_source / "fridge/odFridgeObjects/annotations",
        images_dir=samples_source / "fridge/odFridgeObjects/images",
        class_map=fridge_class_map,
    )

    data_splitter = RandomSplitter([0.8, 0.2])
    train_records, valid_records = parser.parse(data_splitter)

    tfms_ = tfms.A.Adapter([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize()])

    train_ds = Dataset(train_records[:2], tfms_)
    valid_ds = Dataset(valid_records[:2], tfms_)

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


@pytest.fixture(scope="session")
def voc_class_map():
    classes = sorted(
        {
            "person",
            "bird",
            "cat",
            "cow",
            "dog",
            "horse",
            "sheep",
            "aeroplane",
            "bicycle",
            "boat",
            "bus",
            "car",
            "motorbike",
            "train",
            "bottle",
            "chair",
            "diningtable",
            "pottedplant",
            "sofa",
            "tvmonitor",
        }
    )

    return ClassMap(classes=classes)


@pytest.fixture(scope="session")
def fridge_class_map():
    classes = sorted({"milk_bottle", "carton", "can", "water_bottle"})
    return ClassMap(classes)


# COCO fixtures
@pytest.fixture(scope="session")
def coco_dir():
    return Path(__file__).absolute().parent.parent / "samples"


@pytest.fixture(scope="module")
def coco_bbox_parser(coco_dir):
    return parsers.coco(coco_dir / "annotations.json", coco_dir / "images", mask=False)


@pytest.fixture(scope="module")
def coco_mask_parser(coco_dir):
    return parsers.coco(coco_dir / "annotations.json", coco_dir / "images", mask=True)


@pytest.fixture(scope="module")
def coco_keypoints_parser(coco_dir):
    return parsers.COCOKeyPointsParser(
        coco_dir / "keypoints_annotations.json", coco_dir / "images"
    )


@pytest.fixture(scope="module")
def coco_mask_records(coco_mask_parser, coco_imageid_map):
    return coco_mask_parser.parse(
        data_splitter=SingleSplitSplitter(), idmap=coco_imageid_map
    )[0]


@pytest.fixture
def coco_record(coco_mask_records):
    return coco_mask_records[0].copy()


@pytest.fixture
def coco_sample(coco_record):
    return coco_record.load().copy()


@pytest.fixture
def keypoints_img_128372():
    return [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        406,
        218,
        1,
        423,
        218,
        1,
        394,
        235,
        2,
        431,
        236,
        2,
        384,
        257,
        2,
        437,
        269,
        2,
        396,
        281,
        2,
        440,
        292,
        2,
        402,
        292,
        2,
        423,
        293,
        2,
        401,
        330,
        2,
        423,
        329,
        2,
        399,
        370,
        2,
        421,
        371,
        2,
    ]


# VIA fixtures
@pytest.fixture(scope="session")
def via_bbox_class_map():
    classes = sorted({"z", "c", "n", "o"})

    return ClassMap(classes=classes)


@pytest.fixture(scope="session")
def via_dir():
    return Path(__file__).absolute().parent.parent / "samples/via"


# OCHumanKeypoints fixtures
class OCHumanKeypointsMetadata(KeypointsMetadata):
    labels = (
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "right_hip",
        "right_knee",
        "right_ankle",
        "left_hip",
        "left_knee",
        "left_ankle",
        "head",
        "neck",
        "right_ear",
        "left_ear",
        "nose",
        "right_eye",
        "left_eye",
    )


@pytest.fixture(scope="module")
def ochuman_ds(samples_source) -> Tuple[Dataset, Dataset]:
    class OCHumanParser(
        parsers.Parser,
        parsers.FilepathMixin,
        parsers.KeyPointsMixin,
        parsers.LabelsMixin,
        parsers.BBoxesMixin,
    ):
        def __init__(self, annotations_filepath, img_dir):
            self.annotations_dict = json.loads(Path(annotations_filepath).read_bytes())
            self.img_dir = Path(img_dir)
            super().__init__()

        def __iter__(self):
            yield from self.annotations_dict["images"]

        def __len__(self):
            return len(self.annotations_dict["images"])

        def imageid(self, o):
            return int(o["image_id"])

        def filepath(self, o):
            return self.img_dir / o["file_name"]

        def keypoints(self, o):
            return [
                KeyPoints.from_xyv(kps["keypoints"], OCHumanKeypointsMetadata)
                for kps in o["annotations"]
                if kps["keypoints"] is not None
            ]

        def image_width_height(self, o) -> Tuple[int, int]:
            return get_image_size(self.filepath(o))

        def labels(self, o) -> List[Hashable]:
            return [
                "person" for ann in o["annotations"] if ann["keypoints"] is not None
            ]

        def bboxes(self, o) -> List[BBox]:
            return [
                BBox.from_xyxy(*ann["bbox"])
                for ann in o["annotations"]
                if ann["keypoints"] is not None
            ]

    parser = OCHumanParser(
        samples_source / "ochuman/annotations/ochuman.json",
        samples_source / "ochuman/images/",
    )
    train_records, valid_records = parser.parse(
        data_splitter=RandomSplitter([0.8, 0.2])
    )

    presize = 64
    size = 32

    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])
    train_tfms = tfms.A.Adapter(
        [*tfms.A.aug_tfms(size=size, presize=presize, crop_fn=None), tfms.A.Normalize()]
    )

    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    return train_ds, valid_ds


@pytest.fixture()
def ochuman_keypoints_dls(ochuman_ds) -> Tuple[DataLoader, DataLoader]:
    train_ds, valid_ds = ochuman_ds
    train_dl = keypoint_rcnn.train_dl(train_ds, batch_size=2)
    valid_dl = keypoint_rcnn.valid_dl(valid_ds, batch_size=2)

    return train_dl, valid_dl
