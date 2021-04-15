import pytest

from icevision.all import *
from icevision.metrics.confusion_matrix.confusion_matrix_utils import *


def record_template():
    record = BaseRecord(
        (
            SizeRecordComponent(),
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )

    class_map = ClassMap(["a", "b", "c"])
    record.set_record_id(0)
    record.set_filepath("none")
    record.set_img_size(ImgSize(700, 700), original=True)
    record.detection.set_class_map(class_map)
    return record


@pytest.fixture()
def target_bboxes():
    return [
        BBox.from_xywh(100, 100, 300, 300),
        BBox.from_xywh(300, 100, 300, 300),
        BBox.from_xywh(700, 200, 200, 300),
    ]


@pytest.fixture()
def predicted_bboxes():
    return [
        BBox.from_xywh(100, 100, 300, 300),
        BBox.from_xywh(190, 100, 320, 300),
        BBox.from_xywh(400, 200, 200, 300),
    ]


@pytest.fixture()
def predicted_bboxes_wrong():
    return [
        BBox.from_xywh(10, 10, 200, 200),
        BBox.from_xywh(20, 20, 200, 200),
        BBox.from_xywh(10, 10, 200, 200),
    ]


@pytest.fixture
def target(target_bboxes):
    record = record_template()
    record.detection.set_labels_by_id([1, 2, 2])
    record.detection.set_bboxes(target_bboxes)
    return record


@pytest.fixture()
def prediction(predicted_bboxes, predicted_bboxes_wrong):
    pred = record_template()

    pred.add_component(ScoresRecordComponent())
    pred.detection.set_labels_by_id([1, 2, 2, 1, 0, 2])
    pred.detection.set_bboxes(predicted_bboxes + predicted_bboxes_wrong)
    pred.detection.set_scores([0.8, 0.7, 0.5, 0.6, 0.2, 0.4])
    return pred


@pytest.fixture()
def empty_prediction():
    pred = record_template()
    pred.add_component(ScoresRecordComponent())
    pred.detection.set_labels_by_id([])
    pred.detection.set_bboxes([])
    pred.detection.set_scores([])
    return pred


def test_zeroify():
    t = torch.tensor([0.1, 0.0, -10.0, 1000])
    expected_result_0 = torch.tensor([0.1, 0.0, 0.0, 1000])
    expected_result_1 = torch.tensor([0.0, 0.0, 0.0, 0.0])
    assert torch.equal(zeroify_items_below_threshold(t, 0.0), expected_result_0)
    assert torch.equal(zeroify_items_below_threshold(t, -1.0), expected_result_0)
    assert torch.equal(zeroify_items_below_threshold(t, 1000), expected_result_1)


def test_pairwise_iou_empty(target, empty_prediction):
    result = pairwise_iou_record_record(target=target, prediction=empty_prediction)
    empty_result = pairwise_iou_record_record(
        prediction=empty_prediction, target=empty_prediction
    )
    assert result.numel() == 0
    assert result.shape == (0, 3)
    assert empty_result.numel() == 0
    assert empty_result.shape == (0, 0)


def test_pairwise_iou_matching(target, prediction):
    result = pairwise_iou_record_record(target=target, prediction=prediction)
    expected_result = torch.tensor(
        [
            [1.0000, 0.2000, 0.0000],
            [0.5122, 0.5122, 0.0000],
            [0.0000, 0.3636, 0.0000],
            [0.1026, 0.0000, 0.0000],
            [0.1246, 0.0000, 0.0000],
            [0.1026, 0.0000, 0.0000],
        ]
    )
    assert torch.allclose(result, expected_result, atol=1e-4)


def test_match_prediction(target, prediction):
    result = match_records(target, prediction, iou_threshold=0.5)
    expected_result = [
        [
            {"target_bbox": BBox.from_xyxy(100, 100, 400, 400), "target_label": "a"},
            [
                {
                    "predicted_bbox": BBox.from_xyxy(100, 100, 400, 400),
                    "predicted_label": "a",
                    "score": 0.8,
                    "iou_score": 1.0,
                },
                {
                    "predicted_bbox": BBox.from_xyxy(190, 100, 510, 400),
                    "predicted_label": "b",
                    "score": 0.7,
                    "iou_score": 0.5122,
                },
            ],
        ],
        [
            {"target_bbox": BBox.from_xyxy(300, 100, 600, 400), "target_label": "b"},
            [
                {
                    "predicted_bbox": BBox.from_xyxy(190, 100, 510, 400),
                    "predicted_label": "b",
                    "score": 0.7,
                    "iou_score": 0.5122,
                },
            ],
        ],
        [{"target_bbox": BBox.from_xyxy(700, 200, 900, 500), "target_label": "b"}, []],
    ]
    assert result == expected_result


# TODO: add tests for confusion matrix based on pure record matching
