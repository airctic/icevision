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
            {
                "target_bbox": BBox.from_xyxy(100, 100, 400, 400),
                "target_label": "a",
                "target_label_id": 1,
            },
            [
                {
                    "predicted_bbox": BBox.from_xyxy(100, 100, 400, 400),
                    "predicted_label": "a",
                    "predicted_label_id": 1,
                    "score": 0.8,
                    "iou_score": 1.0,
                },
                {
                    "predicted_bbox": BBox.from_xyxy(190, 100, 510, 400),
                    "predicted_label": "b",
                    "predicted_label_id": 2,
                    "score": 0.7,
                    "iou_score": 0.5122,
                },
            ],
        ],
        [
            {
                "target_bbox": BBox.from_xyxy(300, 100, 600, 400),
                "target_label": "b",
                "target_label_id": 2,
            },
            [
                {
                    "predicted_bbox": BBox.from_xyxy(190, 100, 510, 400),
                    "predicted_label": "b",
                    "predicted_label_id": 2,
                    "score": 0.7,
                    "iou_score": 0.5122,
                },
            ],
        ],
        [
            {
                "target_bbox": BBox.from_xyxy(700, 200, 900, 500),
                "target_label": "b",
                "target_label_id": 2,
            },
            [],
        ],
    ]
    assert result == expected_result


@pytest.mark.parametrize(
    "record, prediction_record",
    [
        ("target", "prediction"),
        ("target", "empty_prediction"),
        ("empty_prediction", "prediction"),
        ("empty_prediction", "empty_prediction"),
    ],
)
def test_confusion_matrix_logic(record, prediction_record, request):
    record = request.getfixturevalue(record)
    prediction_record = request.getfixturevalue(prediction_record)

    confusion_matrix = SimpleConfusionMatrix()
    predictions = [Prediction(prediction_record, record)]
    confusion_matrix.accumulate(predictions)
    confusion_matrix.finalize()
    cm = confusion_matrix.confusion_matrix
    assert isinstance(cm, np.ndarray)


def test_confusion_matrix_value(records, preds):
    confusion_matrix = SimpleConfusionMatrix()
    predictions = [
        Prediction(prediction_record, record)
        for prediction_record, record in zip(preds, records)
    ]
    confusion_matrix.accumulate(predictions)
    dummy_result = confusion_matrix.finalize()
    cm_result = confusion_matrix.confusion_matrix
    expected_result = np.diagflat([0, 0, 2, 1])
    assert dummy_result["dummy_value_for_fastai"] == -1
    assert np.equal(cm_result, expected_result).all()
