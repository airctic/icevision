from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.hub.pennfundan import *
from mantisshrimp.engines.fastai import *

### Unified setup ###
source = get_pennfundan_data()
parser = PennFundanParser(source)
splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)
train_dataset = Dataset(train_records)
valid_dataset = Dataset(valid_records)
model = MantisMaskRCNN(num_classes=2)
train_dataloader = model.dataloader(train_dataset, batch_size=2, num_workers=2)
valid_dataloader = model.dataloader(valid_dataset, batch_size=2, num_workers=2)
metric = COCOMetric(valid_records, bbox=True, mask=True)
###

### Engine dependent ###
learn = rcnn_learner(
    dls=[train_dataloader, valid_dataloader], model=model, metrics=[metric]
)

learn.fine_tune(3, lr=2e-4)
###

# TODO: add some tests
# check that model_splits is freezing the correct layers
learn.freeze()
requires_grads = [param.requires_grad for param in learn.model.parameters()]
assert not requires_grads[0]
assert requires_grads[-1]

learn.unfreeze()
requires_grads = [param.requires_grad for param in learn.model.parameters()]
assert requires_grads[0]
assert requires_grads[-1]
