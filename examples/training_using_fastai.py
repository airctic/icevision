"""
Example showing how to train the PETS dataset using Mantisshrimp and [fastai2](https://github.com/fastai/fastai2) library
"""


from mantisshrimp.imports import *
from mantisshrimp import *
import albumentations as A

# import fastai engine provided by the mantisshrimp modules
from mantisshrimp.engines.fastai import *

# Load the PETS dataset
path = datasets.pets.load()

# split dataset lists
data_splitter = RandomSplitter([.8, .2])

# PETS parser: provided out-of-the-box
parser = datasets.pets.parser(path)
train_records, valid_records = parser.parse(data_splitter)

# For convenience
CLASSES = datasets.pets.CLASSES

# shows images with corresponding labels and boxes
show_records(train_records[:6], ncols=3, classes=CLASSES)

# Create both training and validation datasets - using Albumentations transforms out of the box
train_ds = Dataset(train_records, train_albumentations_tfms_pets() )
valid_ds = Dataset(valid_records, valid_albumentations_tfms_pets())

# Create both training and validation dataloaders
train_dl = model.dataloader(train_ds, batch_size=16, num_workers=4, shuffle=True)
valid_dl = model.dataloader(valid_ds, batch_size=16, num_workers=4, shuffle=False)

# Create model
model = faster_rcnn.model(num_classes= len(CLASSES))

# Training the model using fastai2
learn = faster_rcnn.fastai.learner(dls=[train_dl, valid_dl], model=model)
learn.fine_tune(10, lr=1e-4)