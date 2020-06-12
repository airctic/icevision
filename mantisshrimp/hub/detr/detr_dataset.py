__all__ = ["DetrDataset"]

from mantisshrimp.imports import *
from mantisshrimp import *
from PIL import Image

torchvision.datasets.CocoDetection


class DetrDataset(Dataset):
    def __getitem__(self, i):
        record = self.prepare_record(self.records[i])
        x = Image.fromarray(record["img"])
        y = {}
        w, h = x.size
        # image information
        y["image_id"] = tensor(record["imageid"], dtype=torch.int64)
        y["size"] = tensor([h, w], dtype=torch.int64)
        y["orig_size"] = tensor([record["height"], record["width"]], dtype=torch.int64)
        # annotations
        y["labels"] = tensor(record["label"], dtype=torch.int64)
        y["area"] = tensor([o.area for o in record["bbox"]], dtype=torch.float32)
        if "masks" in record:
            y["masks"] = tensor(record["mask"].data, dtype=torch.bool)
        if "iscrowd" in record:
            y["iscrowd"] = tensor(record["iscrowd"], dtype=torch.int64)
        bboxes = [o.xyxy for o in record["bbox"]]
        y["boxes"] = tensor(bboxes, dtype=torch.float32)
        # Apply transforms
        if self.tfm is not None:
            x, y = self.tfm(x, y)
        return x, y
