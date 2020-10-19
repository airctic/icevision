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
        tfm: Transform = None,
    ):
        self.records = records
        self.tfm = tfm

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        data = self.records[i].load().as_dict()
        if self.tfm is not None:
            data = self.tfm(data)
        return data

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self.records)} items>"

    @classmethod
    def from_images(cls, images: Sequence[np.array], tfm: Transform = None):
        """Creates a `Dataset` from a list of images.

        # Arguments
            images: `Sequence` of images in memory (numpy arrays).
            tfm: Transforms to be applied to each item.
        """
        Record = create_mixed_record((ImageRecordMixin,))
        records = []
        for i, image in enumerate(images):
            record = Record()
            record.set_imageid(i)
            record.set_img(image)
            records.append(record)

        return cls(records=records, tfm=tfm)
