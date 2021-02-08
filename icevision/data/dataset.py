__all__ = ["Dataset"]

from icevision.imports import *
from icevision.core import *
from icevision.tfms import *


class Dataset:
    """Container for a list of records and transforms.

    Steps each time an item is requested (normally via directly indexing the `Dataset`):
        * Grab a record from the internal list of records.
        * Prepare the record (open the image, open the mask, add metadata).
        * Apply transforms to the record.

    # Arguments
        records: A list of records.
        tfm: Transforms to be applied to each item.
    """

    def __init__(
        self,
        records: List[dict],
        tfm: Optional[Transform] = None,
    ):
        self.records = records
        self.tfm = tfm(records[0].components_cls) if tfm is not None else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i].load()
        if self.tfm is not None:
            record = self.tfm(record)
        return record

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.records)} items>"

    @classmethod
    def from_images(cls, images: Sequence[np.array], tfm: Transform = None):
        """Creates a `Dataset` from a list of images.

        # Arguments
            images: `Sequence` of images in memory (numpy arrays).
            tfm: Transforms to be applied to each item.
        """
        records = []
        for i, image in enumerate(images):
            record = BaseRecord((ImageRecordComponent,))
            record.set_imageid(i)
            record.set_img(image)
            records.append(record)

        return cls(records=records, tfm=tfm)
