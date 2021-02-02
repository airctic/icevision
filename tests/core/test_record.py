import pytest
from icevision.all import *


@pytest.fixture()
def record(samples_source):
    record = BaseRecord(
        (
            BBoxesRecordComponent,
            LabelsRecordComponent,
            MasksRecordComponent,
            FilepathRecordComponent,
        )
    )

    record.set_imageid(1)
    record.set_image_size(3, 3)
    record.add_labels([1, 2])
    record.add_bboxes([BBox.from_xyxy(1, 2, 4, 4), BBox.from_xyxy(1, 2, 1, 3)])
    record.set_filepath(samples_source / "voc/JPEGImages/2007_000063.jpg")
    mask_filepath = samples_source / "voc/SegmentationObject/2007_000063.png"
    record.add_masks([VocMaskFile(mask_filepath)])

    return record


def test_record_load(record):
    record_loaded = record.load()

    assert isinstance(record_loaded.img, np.ndarray)
    assert isinstance(record_loaded.masks, MaskArray)

    # test original record is not modified
    assert not hasattr(record, "img")
    assert isinstance(record.masks, EncodedRLEs)


@pytest.fixture
def record_empty_annotations():
    record = BaseRecord((BBoxesRecordComponent, LabelsRecordComponent))
    record.set_imageid(2)
    record.set_image_size(3, 3)
    return record


@pytest.fixture
def record_invalid_path():
    record = BaseRecord((FilepathRecordComponent,))

    record.set_imageid(2)
    record.set_image_size(3, 3)
    record.set_filepath("none.jpg")
    return record


@pytest.fixture()
def record_wrong_num_annotations(samples_source):
    record = BaseRecord(
        (BBoxesRecordComponent, LabelsRecordComponent, MasksRecordComponent)
    )

    record.set_imageid(3)
    record.set_image_size(3, 3)
    record.add_labels([1, 2])
    record.add_bboxes([BBox.from_xyxy(1, 2, 4, 4)])
    mask_filepath = samples_source / "voc/SegmentationObject/2007_000063.png"
    record.add_masks([VocMaskFile(mask_filepath)])

    return record


class TestKeypointsMetadata(KeypointsMetadata):
    labels = ("nose", "ankle")


@pytest.fixture()
def record_keypoints():
    record = BaseRecord((BBoxesRecordComponent, KeyPointsRecordComponent))
    record.add_keypoints(
        [
            KeyPoints.from_xyv([0, 0, 0, 1, 1, 1, 2, 2, 2], TestKeypointsMetadata),
            KeyPoints.from_xyv([0, 0, 0], TestKeypointsMetadata),
        ]
    )
    record.add_bboxes([BBox.from_xyxy(1, 2, 4, 4)])

    return record


def test_record_keypoints(record_keypoints):
    assert len(record_keypoints.keypoints) == 2
    assert record_keypoints.keypoints[1].n_visible_keypoints == 0
    assert (record_keypoints.keypoints[0].visible == np.array([0, 1, 2])).all()
    assert (record_keypoints.keypoints[0].x == np.array([0, 1, 2])).all()
    assert record_keypoints.bboxes == [BBox.from_xyxy(1, 2, 4, 4)]
    assert (
        record_keypoints.keypoints[1].y
        == KeyPoints.from_xyv([0, 0, 0], TestKeypointsMetadata).y
    ).all()
    assert record_keypoints.keypoints[0].metadata == TestKeypointsMetadata


def test_record_num_annotations(record):
    assert record.num_annotations() == {"labels": 2, "bboxes": 2, "masks": 2}


def test_record_wrong_num_annotations(record_wrong_num_annotations):
    with pytest.raises(AutofixAbort):
        record_wrong_num_annotations.check_num_annotations()


def test_record_autofix(record):
    success_dict = record.autofix()

    assert record.labels == [1]
    assert record.bboxes == [BBox.from_xyxy(1, 2, 3, 3)]
    assert len(record.masks) == 1


def test_record_autofix_invalid_path(record_invalid_path):
    with pytest.raises(AutofixAbort):
        record_invalid_path.autofix()


def test_autofix_records(
    record, record_invalid_path, record_wrong_num_annotations, record_empty_annotations
):
    records = [
        record,
        record_invalid_path,
        record_wrong_num_annotations,
        record_empty_annotations,
    ]
    records = autofix_records(records)

    assert len(records) == 2
    record = records[0]
    assert record.labels == [1]
    assert record.bboxes == [BBox.from_xyxy(1, 2, 3, 3)]
    assert len(record.masks) == 1
