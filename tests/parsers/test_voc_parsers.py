import pytest
from mantisshrimp import *


@pytest.fixture()
def voc_category2id():
    return {cls: i for i, cls in enumerate(datasets.voc.CLASSES)}


def test_voc_annotation_parser(samples_source, voc_category2id):
    annotation_parser = datasets.voc.VocXmlParser(
        annotations_dir=samples_source / "voc/Annotations",
        images_dir=samples_source / "voc/JPEGImages",
        classes=datasets.voc.CLASSES,
    )
    records = annotation_parser.parse()[0]

    assert len(records) == 2

    record = records[0]
    expected = {
        "imageid": 0,
        "filepath": samples_source / "voc/JPEGImages/2011_003353.jpg",
        "height": 500,
        "width": 375,
        "labels": [voc_category2id["person"]],
        "bboxes": [BBox.from_xyxy(130, 45, 375, 470)],
    }
    assert record == expected

    record = records[1]
    expected = {
        "imageid": 1,
        "filepath": samples_source / "voc/JPEGImages/2007_000063.jpg",
        "width": 500,
        "height": 375,
        "labels": [voc_category2id[k] for k in ["dog", "chair"]],
        "bboxes": [BBox.from_xyxy(123, 115, 379, 275), BBox.from_xyxy(75, 1, 428, 375)],
    }
    assert record == expected


def test_voc_mask_parser(samples_source):
    mask_parser = datasets.voc.VocMaskParser(
        masks_dir=samples_source / "voc/SegmentationClass"
    )
    records = mask_parser.parse()[0]

    record = records[0]
    expected = {
        "imageid": 0,
        "masks": [
            VocMaskFile(samples_source / "voc/SegmentationClass/2007_000063.png"),
        ],
    }
    assert record == expected


def test_voc_combined_parser(samples_source, voc_category2id):
    annotation_parser = datasets.voc.VocXmlParser(
        annotations_dir=samples_source / "voc/Annotations",
        images_dir=samples_source / "voc/JPEGImages",
        classes=datasets.voc.CLASSES,
    )
    mask_parser = datasets.voc.VocMaskParser(
        masks_dir=samples_source / "voc/SegmentationClass"
    )

    combined_parser = CombinedParser(annotation_parser, mask_parser)
    records = combined_parser.parse()[0]

    assert len(records) == 1

    record = records[0]
    expected = {
        "imageid": 1,
        "filepath": samples_source / "voc/JPEGImages/2007_000063.jpg",
        "width": 500,
        "height": 375,
        "labels": [voc_category2id[k] for k in ["dog", "chair"]],
        "bboxes": [BBox.from_xyxy(123, 115, 379, 275), BBox.from_xyxy(75, 1, 428, 375)],
        "masks": [
            VocMaskFile(samples_source / "voc/SegmentationClass/2007_000063.png")
        ],
    }
    assert record == expected
