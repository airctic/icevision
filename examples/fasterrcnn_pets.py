from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.models.rcnn import faster_rcnn
import albumentations as A

# Load the PETS dataset
path = datasets.pets.load()

# split dataset lists
data_splitter = RandomSplitter([0.8, 0.2])

# PETS parser: provided out-of-the-box
parser = datasets.pets.parser(path)
train_records, valid_records = parser.parse(data_splitter)

# For convenience
CLASSES = datasets.pets.CLASSES

# shows images with corresponding labels and boxes
records = train_records[:6]
# show_records(records, ncols=3, classes=CLASSES)

# ImageNet stats
imagenet_mean, imagenet_std = IMAGENET_STATS

# Transform: supporting albumentations transforms out of the box
# Transform for the train dataset
train_tfms = AlbuTransform(
    [
        A.LongestMaxSize(384),
        A.RandomSizedBBoxSafeCrop(320, 320, p=0.3),
        A.HorizontalFlip(),
        A.ShiftScaleRotate(rotate_limit=20),
        A.RGBShift(always_apply=True),
        A.RandomBrightnessContrast(),
        A.Blur(blur_limit=(1, 3)),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

# Transform for the validation dataset
valid_tfms = AlbuTransform(
    [A.LongestMaxSize(384), A.Normalize(mean=imagenet_mean, std=imagenet_std),]
)

# Create both training and validation datasets
train_ds = Dataset(train_records[:100], train_tfms)
valid_ds = Dataset(valid_records[:100], valid_tfms)

# Create both training and validation dataloaders
train_dl = faster_rcnn.train_dataloader(
    train_ds, batch_size=8, num_workers=4, shuffle=True
)
valid_dl = faster_rcnn.valid_dataloader(
    valid_ds, batch_size=8, num_workers=4, shuffle=False
)

# Create model
backbone = faster_rcnn.backbones.resnet34(False)
model = faster_rcnn.model(num_classes=len(CLASSES), backbone=backbone)

# Training the model using fastai2 (be sure that fastai2 is installed)
learn = faster_rcnn.fastai.learner(dls=[train_dl, valid_dl], model=model)

learn.fine_tune(2, lr=1e-4)

# Train model with ligtning
class LightModel(faster_rcnn.lightning.ModelAdapter):
    def configure_optimizers(self):
        sgd = SGD(self.parameters(), lr=1e-4)
        return sgd


light_model = LightModel(model)
trainer = pl.Trainer(max_epochs=2, gpus=1)
trainer.fit(model=light_model, train_dataloader=train_dl, val_dataloaders=valid_dl)


# For inference, create a dataloader with only the images, for simplicity
# lets grab the images from the validation set
model.cuda()
images = [record["img"] for record in valid_ds]
infer_dl = faster_rcnn.infer_dataloader(dataset=images, batch_size=2)

preds = [faster_rcnn.predict(model=model, batch=batch) for batch in infer_dl]
