import collections

from icevision.imports import *
from icevision import BBox, BaseRecord


@dataclass(frozen=True)
class ObjectDetectionItem:
    bbox: BBox
    label: str
    label_id: int
    record_id: int
    item_id: int

    # def __eq__(self, other):
    #     return (
    #         self.record_id == other.record_id
    #         and self.item_id == other.item_id
    #         and self.__class__ == other.__class__
    #     )


@dataclass(frozen=True)
class ObjectDetectionTarget(ObjectDetectionItem):
    # matches: Collection[ObjectDetectionItem] = dataclasses.field(default_factory=list)
    pass


@dataclass(frozen=True)
class ObjectDetectionPrediction(ObjectDetectionItem):
    score: float = None


@dataclass(frozen=True)
class ObjectDetectionMatch(ObjectDetectionItem):
    item: ObjectDetectionItem
    iou_score: float


class Register(collections.UserDict):
    def __init__(self, allow_duplicates=False):
        super().__init__()
        self.allow_duplicates = allow_duplicates
        # self.data = defaultdict(dict)

    def update(self, other=(), /, **kwds):
        self.data.update(other, **kwds)

    def __setitem__(self, key, value: Dict):
        # TODO: change self.data to defaultdict with duplicates or not and move this logic to the inner dict
        duplicate_items = set()
        for item, iou in value.items():
            if not self.allow_duplicates:
                for existing_key, matches_dict in self.data.items():
                    if item in matches_dict.keys():
                        existing_iou = matches_dict[item]
                        if existing_iou >= iou:
                            duplicate_items.add(item)
                        else:
                            del self.data[existing_key][item]
        for item in duplicate_items:
            del value[item]
        # allow multiple matches per prediction
        self[key].update(value)
        # single, best iou match per prediction:
        if len(self.data[key].values()) > 1:
            self.data[key] = dict([max(self[key].items(), key=lambda item: item[1])])

    def filter_by_label_id(self, label_id: int):
        register_view = Register(allow_duplicates=self.allow_duplicates)
        for key, matches_dict in self.data.items():
            if key.label_id == label_id:
                register_view.data[key] = {}
                for inner_key, iou in matches_dict.items():
                    if inner_key.label_id == label_id:
                        register_view.update({key: {inner_key: iou}})
        return register_view

    def register_keys(self, prediction_list):
        self.data = {prediction: {} for prediction in prediction_list}


def default_item():
    return (
        ObjectDetectionPrediction(
            bbox=BBox.from_xyxy(0, 0, 0, 0),
            score=0.0,
            label_id=0,
            label="background",
            record_id=-1,
            item_id=-1,
        ),
        0.0,
    )


def get_best_score_match(prediction_items: Dict[ObjectDetectionItem, float]):
    # fill with dummy if list of prediction_items is empty
    best_item = max(
        prediction_items.items(), key=lambda x: x[0].score, default=default_item()
    )
    return best_item


def get_best_iou_match(prediction_items: Dict[ObjectDetectionItem, float]):
    # fill with dummy if list of prediction_items is empty
    best_item = max(
        prediction_items.items(), key=lambda x: x[1], default=default_item()
    )
    return best_item


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


def pairwise_iou_list_list(
    target_list: List[ObjectDetectionItem], prediction_list: List[ObjectDetectionItem]
) -> torch.Tensor:
    """
    Calculates pairwise iou on prediction and target BaseRecord. Uses torchvision implementation of `box_iou`.
    """
    stacked_preds = [prediction.bbox.to_tensor() for prediction in prediction_list]
    stacked_preds = torch.stack(stacked_preds) if stacked_preds else torch.empty(0, 4)

    stacked_targets = [target.bbox.to_tensor() for target in target_list]
    stacked_targets = (
        torch.stack(stacked_targets) if stacked_targets else torch.empty(0, 4)
    )
    return torchvision.ops.box_iou(stacked_preds, stacked_targets)


def build_target_list(target: BaseRecord) -> List:
    record_id = target.record_id
    target_list = [
        ObjectDetectionTarget(
            record_id=record_id,
            item_id=item_id,
            bbox=bbox,
            label=label,
            label_id=label_id,
        )
        for item_id, (bbox, label, label_id) in enumerate(
            zip(
                target.detection.bboxes,
                target.detection.labels,
                target.detection.label_ids,
            )
        )
    ]

    return target_list


def build_prediction_list(prediction: BaseRecord) -> List:
    record_id = prediction.record_id
    prediction_list = [
        ObjectDetectionPrediction(
            record_id=record_id,
            item_id=item_id,
            bbox=bbox,
            label=label,
            label_id=label_id,
            score=score,
        )
        for item_id, (bbox, label, label_id, score) in enumerate(
            zip(
                prediction.detection.bboxes,
                prediction.detection.labels,
                prediction.detection.label_ids,
                prediction.detection.scores,
            )
        )
    ]

    return prediction_list


def match_predictions_to_targets(
    target: BaseRecord, prediction: BaseRecord, iou_threshold: float = 0.5
) -> Collection:
    """
    matches bboxes, labels from targets with their predictions by iou threshold
    """
    # here we get a tensor of indices that match iou criteria (order is (pred_id, target_id))
    iou_table = pairwise_iou_record_record(target=target, prediction=prediction)
    pairs_indices = torch.nonzero(iou_table > iou_threshold)

    target_list = build_target_list(target)
    prediction_list = build_prediction_list(prediction)
    register = Register(allow_duplicates=True)
    register.register_keys(target_list)

    # appending matching predictions to targets
    for pred_id, target_id in pairs_indices:
        single_prediction = prediction_list[pred_id]
        single_target = target_list[target_id]
        # python value casting needs rounding cause otherwise there are 0.69999991 values
        iou_score = round(iou_table[pred_id, target_id].item(), 4)
        register[single_target] = {single_prediction: iou_score}

    return register


def match_targets_to_predictions(
    target: BaseRecord,
    prediction: BaseRecord,
    iou_threshold: float = 0.5,
    use_coco_matching=True,
) -> Collection:
    """
    matches bboxes, labels from targets with their predictions by iou threshold
    """

    target_list = build_target_list(target)
    prediction_list = sorted(
        build_prediction_list(prediction), key=lambda item: item.score, reverse=True
    )
    # here we get a tensor of indices that match iou criteria (order is (pred_id, target_id))
    iou_table = pairwise_iou_list_list(
        target_list=target_list, prediction_list=prediction_list
    )
    pairs_indices = torch.nonzero(iou_table > iou_threshold)
    register = Register(allow_duplicates=False)
    register.register_keys(prediction_list)
    # appending matching targets to predictions
    for pred_id, target_id in pairs_indices:
        single_target = target_list[target_id]
        single_prediction = prediction_list[pred_id]
        if use_coco_matching:
            # match only same class bboxes
            if single_target.label != single_prediction.label:
                continue

        # python value casting needs rounding cause otherwise there are 0.69999991 values
        iou_score = round(iou_table[pred_id, target_id].item(), 4)
        register[single_prediction] = {single_target: iou_score}

    return register


class NoCopyRepeat(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        return x.expand(self.out_channels, -1, -1)
