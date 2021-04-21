from icevision.imports import *
from icevision import BBox, BaseRecord


def zeroify_items_below_threshold(
    iou_scores: torch.Tensor, threshold: float
) -> torch.Tensor:
    return iou_scores * (iou_scores > threshold).byte()


def get_best_score_item(prediction_items: Collection[Dict]):
    # fill with dummy if list of prediction_items is empty
    dummy = dict(
        predicted_bbox=BBox.from_xyxy(0, 0, 0, 0),
        score=1.0,
        iou_score=1.0,
        predicted_label_id=0,
    )
    best_item = max(prediction_items, key=lambda x: x["score"], default=dummy)
    return best_item


def pairwise_bboxes_iou(
    predicted_bboxes: Sequence[BBox], target_bboxes: Sequence[BBox]
):
    """
    Calculates pairwise iou on lists of bounding boxes. Uses torchvision implementation of `box_iou`.
    :param predicted_bboxes:
    :param target_bboxes:
    :return:
    """
    stacked_preds = [bbox.to_tensor() for bbox in predicted_bboxes]
    stacked_preds = torch.stack(stacked_preds) if stacked_preds else torch.empty(0, 4)

    stacked_targets = [bbox.to_tensor() for bbox in target_bboxes]
    stacked_targets = (
        torch.stack(stacked_targets) if stacked_targets else torch.empty(0, 4)
    )
    return torchvision.ops.box_iou(stacked_preds, stacked_targets)


def pairwise_iou_record_record(target: BaseRecord, prediction: BaseRecord):
    """
    Calculates pairwise iou on prediction and target BaseRecord. Uses torchvision implementation of `box_iou`.
    """
    stacked_preds = [bbox.to_tensor() for bbox in prediction.detection.bboxes]
    stacked_preds = torch.stack(stacked_preds) if stacked_preds else torch.empty(0, 4)

    stacked_targets = [bbox.to_tensor() for bbox in target.detection.bboxes]
    stacked_targets = (
        torch.stack(stacked_targets) if stacked_targets else torch.empty(0, 4)
    )
    return torchvision.ops.box_iou(stacked_preds, stacked_targets)


def match_records(
    target: BaseRecord, prediction: BaseRecord, iou_threshold: float = 0.5
) -> Collection:
    """
    matches bboxes, labels from targets with their predictions by iou threshold
    """
    # here we get a tensor of indices that match iou criteria (order is (pred_id, target_id))
    iou_table = pairwise_iou_record_record(target=target, prediction=prediction)
    pairs_indices = torch.nonzero(iou_table > iou_threshold)

    # creating a list of [target, matching_predictions]
    target_list = [
        [dict(target_bbox=bbox, target_label=label, target_label_id=label_id), []]
        for bbox, label, label_id in zip(
            target.detection.bboxes, target.detection.labels, target.detection.label_ids
        )
    ]
    prediction_list = [
        dict(
            predicted_bbox=bbox,
            predicted_label=label,
            predicted_label_id=label_id,
            score=score,
        )
        for bbox, label, label_id, score in zip(
            prediction.detection.bboxes,
            prediction.detection.labels,
            prediction.detection.label_ids,
            prediction.detection.scores,
        )
    ]

    # appending matches to targets
    for pred_id, target_id in pairs_indices:
        single_prediction = deepcopy(prediction_list[pred_id])
        # python value casting needs rounding cause otherwise there are 0.69999991 values
        iou_score = round(iou_table[pred_id, target_id].item(), 4)
        single_prediction["iou_score"] = iou_score
        # seems like a magic number, but we want to append to the list of target's matching_predictions
        target_list[target_id][1].append(single_prediction)

    return target_list


def add_unknown_labels(ground_truths, predictions, class_map):
    """
    Add Missing labels to the class map by turning gt and preds to sets and then checking which idxs are missing
    :param ground_truths:
    :param predictions:
    :param class_map:
    :return class_map with added missing keys:
    """
    ground_truth_idxs = set(ground_truths)
    prediction_idxs = set(predictions)
    class_map_idxs = set(class_map._class2id.values())
    for missing_idx in ground_truth_idxs.union(prediction_idxs) - class_map_idxs:
        class_map.add_name(f"unknown_id_{missing_idx}")
    return class_map
