__all__ = ["ImgPadStack"]

from icevision.imports import *
from icevision.core import *
from icevision.tfms.batch.batch_transform import BatchTransform


class ImgPadStack(BatchTransform):
    def __init__(self, pad_value: Union[float, Sequence[float]] = 0.0):
        # reshape makes sure array always have one dimension (for the single float case)
        self.pad_value = np.array(pad_value).reshape(-1)

    def apply(self, records: List[RecordType]) -> List[RecordType]:
        max_sizes = np.zeros(3, dtype=int)
        for record in records:
            max_sizes = np.maximum(max_sizes, record["img"].shape)

        img_dtype = records[0]["img"].dtype
        padded_imgs = np.ones((len(records), *max_sizes), dtype=img_dtype)
        padded_imgs *= self.pad_value
        for record, padded_img in zip(records, padded_imgs):
            img = record["img"]

            xmax, ymax, zmax = img.shape
            padded_img[:xmax, :ymax, :zmax] = img
            record["img"] = padded_img

        return records
