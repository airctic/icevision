from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.hub.pennfundan import *
from mantisshrimp.engines.fastai import *
from mantisshrimp.engines.lightning import *

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

### fastai ###
learn = rcnn_learner(
    dls=[train_dataloader, valid_dataloader], model=model, metrics=[metric]
)

learn.fine_tune(3, lr=2e-4)
###

### lightning ###
# class LightModel(RCNNLightningModel):
#     def configure_optimizers(self):
#         opt = SGD(self.parameters(), 2e-4, momentum=0.9)
#         return opt
#
#
# light_model = LightModel(model, metrics=[metric])
# trainer = Trainer(max_epochs=3, gpus=1)
# trainer.fit(light_model, train_dataloader, valid_dataloader)
###
