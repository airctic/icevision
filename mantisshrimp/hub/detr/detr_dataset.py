__all__ = ["DetrDataset"]

from mantisshrimp.imports import *
from mantisshrimp import *


class DetrDataset:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        record = self.dataset[i]
        h, w = record["img"].shape[:-1]
        y = {}
        # image information
        y["image_id"] = tensor(record["imageid"], dtype=torch.int64)
        y["size"] = tensor([h, w], dtype=torch.int64)
        y["orig_size"] = tensor([record["height"], record["width"]], dtype=torch.int64)
        # annotations
        y["labels"] = tensor(record["label"], dtype=torch.int64)
        y["area"] = tensor([o.area for o in record["bbox"]], dtype=torch.float32)
        y["masks"] = tensor(record["mask"].data, dtype=torch.bool)
        y["iscrowd"] = tensor(record["iscrowd"], dtype=torch.int64)
        # TODO: double check we are using the correct coordinates
        bboxes = [o.relative_xcycwh(img_width=w, img_height=h) for o in record["bbox"]]
        y["boxes"] = tensor(bboxes, dtype=torch.float32)
        return y
