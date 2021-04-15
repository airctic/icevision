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
        BBox.from_xywh(700, 200, 200, 300),
    ]


@pytest.fixture()
def predicted_bboxes_wrong():
    return [
        BBox.from_xywh(10, 10, 200, 200),
        BBox.from_xywh(20, 20, 200, 200),
        BBox.from_xywh(10, 10, 200, 200),
    ]


@pytest.fixture
def record(target_bboxes):
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
    pred.detection.set_scores([0.8, 0.7, 0.5, 0.6, 0.2])
    return pred


@pytest.fixture()
def wrong_prediction(predicted_bboxes_wrong):
    pred = record_template()

    pred.add_component(ScoresRecordComponent())
    pred.detection.set_labels_by_id([1, 1, 2])
    pred.detection.set_bboxes(predicted_bboxes_wrong)
    pred.detection.set_scores([0.8, 0.7, 1.0])
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


# todo: refactor below 3 to pairwise record record
def test_pairwise_iou_matching(predicted_bboxes, target_bboxes):
    result = pairwise_bboxes_iou(predicted_bboxes, target_bboxes)
    expected_result = torch.tensor(
        [[1.0000, 0.2000, 0.0000], [0.5122, 0.5122, 0.0000], [0.0000, 0.0000, 1.0000]]
    )
    assert torch.allclose(result, expected_result)
    return result


def test_pairwise_iou_not_matching(predicted_bboxes_wrong, target_bboxes):
    result = pairwise_bboxes_iou(predicted_bboxes_wrong, target_bboxes)
    expected_result = torch.tensor(
        [[0.1026, 0.0000, 0.0000], [0.1246, 0.0000, 0.0000], [0.1026, 0.0000, 0.0000]]
    )
    assert torch.isclose(result, expected_result, 1e-3).all()


def test_pairwise_iou_empty(target_bboxes):
    result = pairwise_bboxes_iou([], target_bboxes)
    empty_result = pairwise_bboxes_iou([], [])
    assert result.numel() == 0
    assert result.shape == (0, 3)
    assert empty_result.numel() == 0
    assert empty_result.shape == (0, 0)


def test_pairwise_iou_record_record(records, preds):
    for prediction, target in zip(preds, records):
        result = pairwise_iou_record_record(prediction=prediction, target=target)
        assert isinstance(result, torch.Tensor)


def test_couple_with_targets(target_bboxes, predicted_bboxes):
    predicted_bboxes = predicted_bboxes

    iou = pairwise_bboxes_iou(
        predicted_bboxes=predicted_bboxes, target_bboxes=target_bboxes
    )
    coupled_list = couple_with_targets(
        predicted_bboxes=predicted_bboxes, iou_scores=iou
    )
    expected_result = [
        [BBox.from_xywh(100, 100, 300, 300), BBox.from_xywh(190, 100, 320, 300)],
        [BBox.from_xywh(100, 100, 300, 300), BBox.from_xywh(190, 100, 320, 300)],
        [BBox.from_xywh(700, 200, 200, 300)],
    ]
    assert len(coupled_list) == 3
    assert [len(preds) for preds in coupled_list] == [2, 2, 1]
    assert coupled_list == expected_result


def test_couple_records(record, prediction):
    predicted_bboxes = record2predictions(prediction)
    target_bboxes = record2targets(record)

    iou = pairwise_bboxes_iou(
        predicted_bboxes=predicted_bboxes, target_bboxes=target_bboxes
    )
    coupled_list = couple_with_targets(
        predicted_bboxes=record2predictions(prediction), iou_scores=iou
    )
    expected_result = [
        [BBox.from_xywh(100, 100, 300, 300), BBox.from_xywh(190, 100, 320, 300)],
        [BBox.from_xywh(100, 100, 300, 300), BBox.from_xywh(190, 100, 320, 300)],
        [BBox.from_xywh(700, 200, 200, 300)],
    ]
    assert len(coupled_list) == 3
    assert [len(preds) for preds in coupled_list] == [2, 2, 1]
    assert coupled_list == expected_result


def test_couple_with_wrong_preds(target_bboxes, predicted_bboxes_wrong):
    iou_scores = pairwise_bboxes_iou(
        predicted_bboxes=predicted_bboxes_wrong, target_bboxes=target_bboxes
    )

    that_match = torch.any(iou_scores > 0.5, dim=1)
    iou_scores = iou_scores[that_match]
    predicted_bboxes_wrong = list(
        itertools.compress(predicted_bboxes_wrong, that_match)
    )

    coupled_list = couple_with_targets(
        predicted_bboxes=predicted_bboxes_wrong, iou_scores=iou_scores
    )
    assert coupled_list == [[], [], []]


def test_match_preds_with_targets(prediction, record):
    targets, matched_preds = match_preds_with_targets(prediction, record)
    assert len(targets) == len(matched_preds)
    assert targets == [
        (BBox.from_xyxy(100, 100, 400, 400), 1),
        (BBox.from_xyxy(300, 100, 600, 400), 2),
        (BBox.from_xyxy(700, 200, 900, 500), 2),
    ]
    assert matched_preds == [
        [
            (BBox.from_xyxy(100, 100, 400, 400), 0.8, 1),
            (BBox.from_xyxy(190, 100, 510, 400), 0.7, 2),
        ],
        [
            (BBox.from_xyxy(190, 100, 510, 400), 0.7, 2),
        ],
        [(BBox.from_xyxy(700, 200, 900, 500), 0.5, 2)],
    ]
