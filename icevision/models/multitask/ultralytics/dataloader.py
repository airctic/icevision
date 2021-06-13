from icevision.imports import *
from icevision.core import *
from icevision.models.utils import *
from icevision.models.ultralytics.yolov5.dataloaders import (
    _build_train_sample as _build_train_detection_sample,
)
from torch import Tensor
from icevision.models.multitask.data.dtypes import *


def build_single_aug_batch(
    records: Sequence[RecordType],
) -> Tuple[TensorList, TensorDict, Sequence[RecordType]]:
    """Builds a batch in the format required by the model when training.

    # Arguments
        records: A `Sequence` of records.

    # Returns
        A tuple with two items. The first will be a tuple like `(images, detection_targets, classification_targets)`
        in the input format required by the model. The second will be an updated list
        of the input records.

    # Examples

    Use the result of this function to feed the model.
    ```python
    batch, records = build_train_batch(records)
    outs = model(*batch)
    ```
    """
    images, detection_targets = [], []
    classification_targets = defaultdict(list)

    for i, record in enumerate(records):
        image, detection_target = _build_train_detection_sample(record)
        images.append(image)

        detection_target[:, 0] = i if detection_target.numel() > 0 else None
        detection_targets.append(detection_target)

        # Classification
        for comp in record.components:
            name = comp.task.name
            if isinstance(comp, ClassificationLabelsRecordComponent):
                if comp.is_multilabel:
                    labels = comp.one_hot_encoded()
                    classification_targets[name].append(labels)
                else:
                    labels = comp.label_ids
                    classification_targets[name].extend(labels)

    classification_targets = {k: tensor(v) for k, v in classification_targets.items()}

    return (
        torch.stack(images, 0),
        torch.cat(detection_targets, 0),
        classification_targets,
    ), records


def build_multi_aug_batch(
    records: Sequence[RecordType], classification_transform_groups: dict
):
    detection_images = []
    detection_targets = []
    classification_data = defaultdict(lambda: defaultdict(list))
    classification_targets = defaultdict(list)

    for i, record in enumerate(records):
        detection_image, detection_target = _build_train_detection_sample(record)
        detection_images.append(detection_image)

        detection_target[:, 0] = i if detection_target.numel() > 0 else None
        detection_targets.append(detection_target)

        for key, group in classification_transform_groups.items():
            task = getattr(record, group["tasks"][0])
            classification_data[key]["tasks"] = group["tasks"]
            classification_data[key]["images"].append(im2tensor(task.img))

        for comp in record.components:
            name = comp.task.name
            if isinstance(comp, ClassificationLabelsRecordComponent):
                if comp.is_multilabel:
                    labels = comp.one_hot_encoded()
                    classification_targets[name].append(labels)
                else:
                    labels = comp.label_ids
                    classification_targets[name].extend(labels)

    # Massage data
    for group in classification_data.values():
        group["targets"] = {
            task: tensor(classification_targets[task]) for task in group["tasks"]
        }
        group["images"] = torch.stack(group["images"])
    classification_data = {k: dict(v) for k, v in classification_data.items()}

    detection_data = dict(
        images=torch.stack(detection_images, 0),
        targets=torch.cat(detection_targets, 0),
    )

    return (detection_data, classification_data)
    # return (detection_data, classification_data), records
    # return dict(detection=detection_data, classification=classification_data)
