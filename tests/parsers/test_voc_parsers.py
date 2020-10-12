from icevision.all import *


def test_voc_annotation_parser(samples_source, voc_class_map):
    annotation_parser = parsers.voc(
        annotations_dir=samples_source / "voc/Annotations",
        images_dir=samples_source / "voc/JPEGImages",
        class_map=voc_class_map,
    )
    records = annotation_parser.parse(data_splitter=SingleSplitSplitter())[0]

    assert len(records) == 2

    record = records[0]
    expected = {
        "imageid": 0,
        "filepath": samples_source / "voc/JPEGImages/2007_000063.jpg",
        "width": 500,
        "height": 375,
        "labels": [voc_class_map.get_name(k) for k in ["dog", "chair"]],
        "bboxes": [BBox.from_xyxy(123, 115, 379, 275), BBox.from_xyxy(75, 1, 428, 375)],
    }
    assert record == expected

    record = records[1]
    expected = {
        "imageid": 1,
        "filepath": samples_source / "voc/JPEGImages/2011_003353.jpg",
        "height": 500,
        "width": 375,
        "labels": [voc_class_map.get_name("person")],
        "bboxes": [BBox.from_xyxy(130, 45, 375, 470)],
    }
    assert record == expected


def test_voc_mask_parser(samples_source, voc_class_map):
    parser = parsers.voc(
        annotations_dir=samples_source / "voc/Annotations",
        images_dir=samples_source / "voc/JPEGImages",
        class_map=voc_class_map,
        masks_dir=samples_source / "voc/SegmentationClass",
    )

    records = parser.parse(data_splitter=SingleSplitSplitter())[0]

    assert len(records) == 1

    record = records[0]
    expected = {
        "imageid": 0,
        "filepath": samples_source / "voc/JPEGImages/2007_000063.jpg",
        "width": 500,
        "height": 375,
        "labels": [voc_class_map.get_name(k) for k in ["dog", "chair"]],
        "bboxes": [BBox.from_xyxy(123, 115, 379, 275), BBox.from_xyxy(75, 1, 428, 375)],
        "masks": [
            VocMaskFile(samples_source / "voc/SegmentationClass/2007_000063.png")
        ],
    }
    assert record == expected
