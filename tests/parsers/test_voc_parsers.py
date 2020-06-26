import pytest
from mantisshrimp.imports import *
from mantisshrimp import *


# source = Path("/home/lgvaz/git/mantisshrimp2/samples/voc")
@pytest.fixture()
def source():
    return Path(__file__).absolute().parent.parent.parent / "samples/voc"


def test_voc_annotation_parser(source):
    annotation_parser = VOCAnnotationParser(
        annotations_dir=source / "Annotations", images_dir=source / "JPEGImages"
    )
    records = annotation_parser.parse()[0]

    assert len(records) == 2
    assert annotation_parser.label_map.i2imageid == ["person", "dog", "chair"]

    record = records[0]
    expected = {
        "imageid": 0,
        "filepath": source / "JPEGImages/2011_003353.jpg",
        "height": 500,
        "width": 375,
        "label": [0],
        "bbox": [BBox.from_xyxy(130, 45, 375, 470)],
    }
    assert record == expected

    record = records[1]
    expected = {
        "imageid": 1,
        "filepath": source / "JPEGImages/2007_000063.jpg",
        "width": 500,
        "height": 375,
        "label": [1, 2],
        "bbox": [BBox.from_xyxy(123, 115, 379, 275), BBox.from_xyxy(75, 1, 428, 375)],
    }
    assert record == expected


def test_voc_mask_parser(source):
    mask_parser = VOCMaskParser(masks_dir=source / "SegmentationClass")
    records = mask_parser.parse()[0]

    record = records[0]
    expected = {
        "imageid": 0,
        "mask": [MaskFile(source / "SegmentationClass/2007_000063.png"),],
    }
    assert record == expected


def test_voc_combined_parser(source):
    annotation_parser = VOCAnnotationParser(
        annotations_dir=source / "Annotations", images_dir=source / "JPEGImages"
    )
    mask_parser = VOCMaskParser(masks_dir=source / "SegmentationClass")

    combined_parser = CombinedParser(annotation_parser, mask_parser)
    records = combined_parser.parse()[0]

    assert len(records) == 1

    record = records[0]
    expected = {
        "imageid": 1,
        "filepath": source / "JPEGImages/2007_000063.jpg",
        "width": 500,
        "height": 375,
        "label": [1, 2],
        "bbox": [BBox.from_xyxy(123, 115, 379, 275), BBox.from_xyxy(75, 1, 428, 375)],
        "mask": [MaskFile(source / "SegmentationClass/2007_000063.png")],
    }
    assert record == expected
