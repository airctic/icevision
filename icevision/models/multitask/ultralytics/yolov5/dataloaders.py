"""
YOLO-V5 dataloaders for multitask training.

The model uses a peculiar format for bounding box annotations where the
  length of the tensor is the total number of bounding boxes for that batch
The first dimension is the index of the image that the box belongs to/
See https://discord.com/channels/735877944085446747/770279401791160400/853691059338084372
  for a more thorough explanation
"""

from icevision.imports import *
from icevision.core import *
from icevision.models.utils import *
from icevision.models.ultralytics.yolov5.dataloaders import (
    _build_train_sample as _build_train_detection_sample,
)
from torch import Tensor
from icevision.models.multitask.utils.dtypes import *


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

        # See file header for more info on why this is done
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
    """
    Docs:
        Take as inputs `records` and `classification_transform_groups` and return
        a tuple of dictionaries, one for detection data and the other for classification.

        See `icevision.models.multitask.data.dataset.HybridAugmentationsRecordDataset`
        for example of what `records` and `classification_transform_groups` look like


    Returns:
        A tuple with two items:
        1. A tuple with two dictionaries - (`detection_data`, `classification_data`)
            `detection_data`:
                {
                    "detection": dict(
                        images: Tensor = ...,  # (N,C,H,W)
                        targets: Tensor = ..., # of shape (num_boxes, 6)
                                               # (img_idx, box_class_idx, **bbox_relative_coords)
                    )
                }
            `classification_data`:
                {
                    "group1": dict(
                        tasks = ["shot_composition"],
                        images: Tensor = ...,
                        targets=dict(
                            "shot_composition": Tensor = ...,
                        )
                    ),
                    "group2": dict(
                        tasks = ["color_saturation", "shot_framing"],
                        images: Tensor = ...,
                        targets=dict(
                            "color_saturation": Tensor = ...,
                            "shot_framing": Tensor = ...,
                        )
                    )
                }
        2. Loaded records (same ones passed as inputs)
    """
    detection_images = []
    detection_targets = []
    classification_data = defaultdict(lambda: defaultdict(list))
    classification_targets = defaultdict(list)

    for i, record in enumerate(records):
        detection_image, detection_target = _build_train_detection_sample(record)
        detection_images.append(detection_image)

        # See file header for more info on why this is done
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

    return (detection_data, classification_data), records
