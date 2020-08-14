"""
Example showing how to train the PETS dataset, showcasing [fastai2](https://github.com/fastai/fastai2) and [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)
"""

# Installing Mantisshrimp
# !pip install git+git://github.com/airctic/mantisshrimp.git#egg=mantisshrimp[all] --upgrade

# Imports
from mantisshrimp.all import *

# Load the PETS dataset
path = datasets.pets.load()

# Get the class_map, a utility that maps from number IDs to classs names
class_map = datasets.pets.class_map()

# Randomly split our data into train/valid
data_splitter = RandomSplitter([0.8, 0.2])

# PETS parser: provided out-of-the-box
parser = datasets.pets.parser(data_dir=path, class_map=class_map)
train_records, valid_records = parser.parse(data_splitter)

# shows images with corresponding labels and boxes
show_records(train_records[:6], ncols=3, class_map=class_map, show=True)

# Define transforms - using Albumentations transforms out of the box
train_tfms = tfms.A.Adapter(
    [*tfms.A.aug_tfms(size=384, presize=512), tfms.A.Normalize()]
)
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])
# Create both training and validation datasets
train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

# Create both training and validation dataloaders
train_dl = faster_rcnn.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

# Create model
model = faster_rcnn.model(num_classes=len(class_map))

# Define metrics
metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

# Train using fastai2
learn = faster_rcnn.fastai.learner(
    dls=[train_dl, valid_dl], model=model, metrics=metrics
)
learn.fine_tune(10, lr=1e-4)

# Train using pytorch-lightning
class LightModel(faster_rcnn.lightning.ModelAdapter):
    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-4)


light_model = LightModel(model, metrics=metrics)

trainer = pl.Trainer(max_epochs=10, gpus=1)
trainer.fit(light_model, train_dl, valid_dl)
