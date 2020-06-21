from fastai2.vision.all import *
from fastai2.metrics import Metric as FastaiMetric
from fastai2.data.load import DataLoader as FastaiDataLoader
from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.hub.pennfundan import *
from mantisshrimp.engines.fastai import *

# Unified setup
source = get_pennfundan_data()
parser = PennFundanParser(source)
splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)
train_dataset = Dataset(train_records)
valid_dataset = Dataset(valid_records)
model = MantisMaskRCNN(num_classes=2)
train_dataloader = model.dataloader(train_dataset, batch_size=2, num_workers=2)
valid_dataloader = model.dataloader(valid_dataset, batch_size=2, num_workers=2)
###

metric = COCOMetric(valid_records, bbox=True, mask=True)
metric = FastaiMetricAdapter(metric)

train_dataloader2 = convert_dataloader_to_fastai(train_dataloader)
valid_dataloader2 = convert_dataloader_to_fastai(valid_dataloader)
# TODO: Check if cuda is available, see how fastai does it
dataloaders = DataLoaders(train_dataloader2, valid_dataloader2).to(torch.device("cuda"))

learn = rcnn_learner(dls=dataloaders, model=model)

learn.fit_one_cycle(3, lr=2e-4)

# TODO: add some tests
# check that model_splits is freezing the correct layers
