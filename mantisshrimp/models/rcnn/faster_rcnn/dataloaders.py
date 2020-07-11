__all__ = ["build_train_batch", "build_valid_batch"]

from mantisshrimp.imports import *
from mantisshrimp.parsers import *


def build_train_batch(
    records: List[RecordType],
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    images, targets = [], []
    for record in records:
        image = im2tensor(record["img"])
        images.append(image)

        target = {
            "labels": tensor(record["labels"], dtype=torch.int64),
            "boxes": tensor(
                [bbox.xyxy for bbox in record["bboxes"]], dtype=torch.float
            ),
        }
        targets.append(target)

    return images, targets


def build_valid_batch(
    records: List[RecordType],
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    return build_train_batch(records=records)
