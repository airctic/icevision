import numpy as np

from icevision import tfms
from icevision.data import Dataset
from icevision.models.ross import efficientdet


def test_mosaic(fridge_ds):
    train_ds, valid_ds = fridge_ds
    union_ds = Dataset(train_ds.records + valid_ds.records, tfm=train_ds.tfm)
    batch_tfms = tfms.batch.Mosaic()
    train_dl = efficientdet.train_dl(union_ds, batch_size=4, batch_tfms=batch_tfms)
    batch = next(iter(train_dl))

    # tfmed_records = tfms.batch.Mosaic()(records)
