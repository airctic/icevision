import imp
from venv import create
import pytest
from PIL import Image
import pytest

fo = pytest.importorskip("fiftyone")

from icevision import tfms
from icevision import data
from icevision.data import Prediction
from icevision.data.convert_records_to_fo import (
    _convert_bbox_to_fo_bbox,
    record_to_fo_detections,
)

# Test subject
from icevision.data import (
    convert_record_to_fo_sample,
    convert_prediction_to_fo_sample,
    create_fo_dataset,
)
from tests.conftest import object_detection_record


# Fixtures
@pytest.fixture(scope="function")
def cleanup_fo_dataset():
    dataset_test_names = ["_iv_test", "_iv_test_1"]
    yield
    for dataset_name in dataset_test_names:
        if dataset_name in fo.list_datasets():
            ds = fo.load_dataset(dataset_name)
            ds.delete()
            del ds


# Test from low to high abstraction
def test_record_to_fo_detections(object_detection_record):
    detections = record_to_fo_detections(
        object_detection_record, lambda bbox: _convert_bbox_to_fo_bbox(bbox, 500, 500)
    )

    for fo_bbox in detections:
        assert isinstance(fo_bbox, fo.Detection)


def test_convert_record_to_fo_sample(object_detection_record):
    # Test create new sample
    test_sample = convert_record_to_fo_sample(
        object_detection_record, "test", None, False, None, lambda x: x
    )
    assert isinstance(test_sample, fo.Sample)
    assert hasattr(test_sample, "test")
    assert len(test_sample["test"]) > 0

    # Test add to existing sample  and autosave
    sample = fo.Sample(object_detection_record.common.filepath)
    test_sample = convert_record_to_fo_sample(
        object_detection_record, "test", sample, False, None, lambda x: x
    )
    assert isinstance(test_sample, fo.Sample)
    assert hasattr(test_sample, "test")
    assert len(test_sample["test"]) > 0

    # Test iv postprocess bbox
    test_tfms = [
        tfms.A.Adapter([*tfms.A.resize_and_pad([1000, 1000]), tfms.A.Normalize()])
    ]
    test_sample = convert_record_to_fo_sample(
        object_detection_record, "test", None, False, test_tfms, None
    )

    assert isinstance(test_sample, fo.Sample)
    assert hasattr(test_sample, "test")
    assert len(test_sample["test"]) > 0


def test_convert_prediction_to_fo_sample(object_detection_record):
    # Create Prediction
    object_detection_record.original_img_size = [1000, 1000]
    object_detection_prediction = Prediction(
        object_detection_record, object_detection_record
    )

    test_sample = convert_prediction_to_fo_sample(
        object_detection_prediction, undo_bbox_tfms_fn=lambda x: x
    )

    del (
        object_detection_record.original_img_size
    )  # Cleanup record, that has scope session

    assert isinstance(test_sample, fo.Sample)

    assert hasattr(test_sample, "ground_truth")
    assert len(test_sample["ground_truth"]) > 0

    assert hasattr(test_sample, "prediction")
    assert len(test_sample["prediction"]) > 0


def test_create_fo_dataset(object_detection_record, cleanup_fo_dataset):
    # Create Prediction
    object_detection_record.original_img_size = [1000, 1000]
    object_detection_prediction = Prediction(
        object_detection_record, object_detection_record
    )

    # Test for record and with existing dataset
    dataset = fo.Dataset("_iv_test")
    dataset = create_fo_dataset(
        [object_detection_prediction], dataset_name="_iv_test", exist_ok=True
    )

    del (
        object_detection_record.original_img_size
    )  # Cleanup record, that has scope session

    assert isinstance(dataset, fo.Dataset)
    assert len(list(dataset.iter_samples())) > 0

    # Test for prediction and create new dataset
    dataset = create_fo_dataset([object_detection_record], "_iv_test_1")

    assert isinstance(dataset, fo.Dataset)
    assert len(list(dataset.iter_samples())) > 0
