import pytest

from icevision import BBox
from icevision.imports import *
from icevision.metrics.confusion_matrix.confusion_matrix_utils import *


@pytest.fixture()
def target_bboxes():
    predicted_bboxes = [
        BBox.from_xywh(100, 100, 300, 300),
        BBox.from_xywh(700, 200, 200, 300),
    ]
    return predicted_bboxes


@pytest.fixture()
def predicted_bboxes():
    predicted_bboxes = [
        DetectedBBox.from_xywh(100, 100, 300, 300, 1.0, 1),
        DetectedBBox.from_xywh(700, 200, 200, 300, 1.0, 2),
    ]
    return predicted_bboxes


@pytest.fixture()
def predicted_bboxes_wrong():
    predicted_bboxes = [
        DetectedBBox.from_xywh(10, 10, 200, 200, 1.0, 1),
        DetectedBBox.from_xywh(20, 20, 200, 200, 1.0, 1),
        DetectedBBox.from_xywh(10, 10, 200, 200, 1.0, 2),
    ]
    return predicted_bboxes


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
    result = pairwise_iou(predicted_bboxes, target_bboxes)
    expected_result = torch.eye(2)
    assert torch.equal(result, expected_result)
    return result


def test_pairwise_iou_not_matching(predicted_bboxes_wrong, target_bboxes):
    result = pairwise_iou(predicted_bboxes_wrong, target_bboxes)
    expected_result = torch.tensor(
        [[0.1026, 0.0000], [0.1246, 0.0000], [0.1026, 0.0000]]
    )
    assert torch.isclose(result, expected_result, 1e-3).all()


def test_empty_iou(empty_detected_bboxes, target_bboxes):
    result = pairwise_iou(empty_detected_bboxes, target_bboxes)
    empty_result = pairwise_iou(empty_detected_bboxes, empty_detected_bboxes)
    assert result.numel() == 0
    assert result.shape == (0, 2)
    assert empty_result.numel() == 0
    assert empty_result.shape == (0, 0)


def test_couple_with_targets(target_bboxes, predicted_bboxes):
    predicted_bboxes = predicted_bboxes * 2
    iou = pairwise_iou(predicted_bboxes=predicted_bboxes, target_bboxes=target_bboxes)
    coupled_list = couple_with_targets(
        predicted_bboxes=predicted_bboxes, iou_scores=iou
    )
    expected_result = [
        [
            DetectedBBox.from_xywh(100, 100, 300, 300, 1.0, 1),
            DetectedBBox.from_xywh(100, 100, 300, 300, 1.0, 1),
        ],
        [
            DetectedBBox.from_xywh(700, 200, 200, 300, 1.0, 2),
            DetectedBBox.from_xywh(700, 200, 200, 300, 1.0, 2),
        ],
    ]
    assert len(coupled_list) == 2
    assert [len(preds) for preds in coupled_list] == [2, 2]
    assert coupled_list == expected_result


def test_couple_with_wrong_preds(target_bboxes, predicted_bboxes_wrong):
    iou_scores = pairwise_iou(
        predicted_bboxes=predicted_bboxes_wrong, target_bboxes=target_bboxes
    )
    coupled_list = couple_with_targets(
        predicted_bboxes=predicted_bboxes_wrong, iou_scores=iou_scores
    )
    that_match = torch.any(iou_scores > self._iou_threshold, dim=1)
    iou_scores = iou_scores[that_match]
