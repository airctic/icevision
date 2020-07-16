__all__ = ["ImgPadStack"]

from mantisshrimp.imports import *
from mantisshrimp.parsers import *
from mantisshrimp.transforms.batch_tfms.batch_transform import BatchTransform


class ImgPadStack(BatchTransform):
    def apply(self, records: List[RecordType]) -> List[RecordType]:
        max_sizes = np.zeros(3, dtype=int)
        for record in records:
            max_sizes = np.maximum(max_sizes, record["img"].shape)

        padded_imgs = np.zeros((len(records), *max_sizes), dtype=np.uint8)
        for record, padded_img in zip(records, padded_imgs):
            img = record["img"]
            record["shape_before_img_pad_stack"] = img.shape

            xmax, ymax, zmax = img.shape
            padded_img[:xmax, :ymax, :zmax] = img
            record["img"] = padded_img

        return records
