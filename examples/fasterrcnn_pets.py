from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.datasets import pets
from mantisshrimp.engines.fastai import *
import albumentations as A

data_dir = pets.load()
parser = pets.parser(data_dir)

splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)

all_records = train_records + valid_records

show_record(train_records[0], show=True)

common_tfms = [A.LongestMaxSize(320)]
aug_tfms = [A.HorizontalFlip()]

train_tfms = AlbuTransform(aug_tfms + common_tfms)
valid_tfms = AlbuTransform(common_tfms)

train_ds = Dataset(train_records, tfm=train_tfms)
valid_ds = Dataset(valid_records, tfm=valid_tfms)

sample = train_ds[0]
show_annotation(
    im=sample["img"], labels=sample["label"], bboxes=sample["bbox"], show=True
)

# TODO: Rethink CATEGORIES, should they always be present in parser?
# Should call it CLASSES instead of CATEGORIES?
model = MantisFasterRCNN(num_classes=len(pets.CATEGORIES))

train_dl = model.dataloader(train_ds, batch_size=2, shuffle=True)
valid_dl = model.dataloader(valid_ds, batch_size=2)

metrics = [COCOMetric(valid_records, bbox=True, mask=False)]

learn = rcnn_learner(dls=[train_dl, valid_dl], model=model)

learn.fit_one_cycle(1, 1e-3)
