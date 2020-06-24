from mantisshrimp import *
from mantisshrimp.hub.pennfundan import *
from mantisshrimp.engines.fastai import *
import albumentations as A

source = get_pennfundan_data()
parser = PennFundanParser(source)

splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)

train_transforms = AlbuTransform([A.Flip()])

train_dataset = Dataset(train_records, train_records)
valid_dataset = Dataset(valid_records)

model = MantisMaskRCNN(num_classes=2)
metric = COCOMetric(valid_records, bbox=True, mask=True)

train_dataloader = model.dataloader(train_dataset, batch_size=2, num_workers=2)
valid_dataloader = model.dataloader(valid_dataset, batch_size=2, num_workers=2)


learn = rcnn_learner(
    dls=[train_dataloader, valid_dataloader], model=model, metrics=[metric]
)

learn.fine_tune(3, lr=2e-4)
