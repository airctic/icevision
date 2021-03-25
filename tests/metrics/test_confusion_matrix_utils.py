import pytest

from icevision.all import *
from icevision.metrics.confusion_matrix.confusion_matrix_utils import *

# from .test_coco_metric import records, preds


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


@pytest.fixture
def record(target_bboxes):
    def _get_record():
        return BaseRecord(
            (
                SizeRecordComponent(),
                FilepathRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
            )
        )

    class_map = ClassMap(["a", "b", "c"])

    record = _get_record()
    record.set_record_id(0)
    record.set_filepath("none")
    record.set_img_size(ImgSize(700, 700), original=True)
    record.detection.set_class_map(class_map)
    record.detection.add_labels_by_id([1, 2, 2])
    record.detection.add_bboxes(target_bboxes)

    return record


@pytest.fixture()
def pred(record, predicted_bboxes, predicted_bboxes_wrong):
    pred = deepcopy(record)
    pred.add_component(ScoresRecordComponent())

    pred = deepcopy(pred)
    pred.detection.set_labels_by_id([1, 2, 2, 1, 0, 2])
    pred.detection.set_bboxes(predicted_bboxes + predicted_bboxes_wrong)
    pred.detection.set_scores([0.8, 0.7, 0.5, 0.6, 0.2])
    return pred


@pytest.fixture()
def predicted_bboxes_wrong():
    return [
        BBox.from_xywh(10, 10, 200, 200),
        BBox.from_xywh(20, 20, 200, 200),
        BBox.from_xywh(10, 10, 200, 200),
    ]


@pytest.fixture()
def wrong_preds(record, predicted_bboxes_wrong):
    pred = deepcopy(record)
    pred.add_component(ScoresRecordComponent())

    pred = deepcopy(pred)
    pred.detection.set_labels_by_id([1, 1, 2])
    pred.detection.set_bboxes(predicted_bboxes_wrong)
    pred.detection.set_scores([0.8, 0.7, 1.0])
    return pred


@pytest.fixture()
def empty_detected_bboxes():
    return []


def test_zeroify():
    t = torch.tensor([0.1, 0.0, -10.0, 1000])
    expected_result_0 = torch.tensor([0.1, 0.0, 0.0, 1000])
    expected_result_1 = torch.tensor([0.0, 0.0, 0.0, 0.0])
    assert torch.equal(zeroify_items_below_threshold(t, 0.0), expected_result_0)
    assert torch.equal(zeroify_items_below_threshold(t, -1.0), expected_result_0)
    assert torch.equal(zeroify_items_below_threshold(t, 1000), expected_result_1)


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


def test_empty_iou(empty_detected_bboxes, target_bboxes):
    result = pairwise_bboxes_iou(empty_detected_bboxes, target_bboxes)
    empty_result = pairwise_bboxes_iou(empty_detected_bboxes, empty_detected_bboxes)
    assert result.numel() == 0
    assert result.shape == (0, 3)
    assert empty_result.numel() == 0
    assert empty_result.shape == (0, 0)


def test_couple_with_targets(target_bboxes, predicted_bboxes, pred):
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


def test_couple_records(record, pred):
    predicted_bboxes = record2predictions(pred)
    target_bboxes = record2targets(record)

    iou = pairwise_bboxes_iou(
        predicted_bboxes=predicted_bboxes, target_bboxes=target_bboxes
    )
    coupled_list = couple_with_targets(
        predicted_bboxes=record2predictions(pred), iou_scores=iou
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


def test_pairwise_predictions_targets_iou(record, pred):
    predictions = record2predictions(pred)
    targets = record2targets(record)
    result = pairwise_iou_predictions_targets(predictions, targets)
    assert isinstance(result, torch.Tensor)


def test_match_preds_with_targets(pred, record):
    targets, matched_preds = match_preds_with_targets(pred, record)
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
