from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.engines.fastai import *
import albumentations as A

data_dir = datasets.pets.load()
parser = datasets.pets.parser(data_dir)

splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)

imagenet_mean, imagenet_std = IMAGENET_STATS
img_size = 256
valid_tfms = AlbuTransform(
    [
        A.Resize(img_size, img_size, always_apply=True),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

train_tfms = AlbuTransform(
    [
        A.LongestMaxSize(img_size),
        # A.RandomSizedBBoxSafeCrop(320, 320, p=0.3),
        A.Resize(img_size, img_size, always_apply=True),
        A.HorizontalFlip(),
        A.ShiftScaleRotate(rotate_limit=20),
        A.RGBShift(always_apply=True),
        A.RandomBrightnessContrast(),
        A.Blur(blur_limit=(1, 3)),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)


train_ds = Dataset(train_records, tfm=train_tfms)
valid_ds = Dataset(valid_records, tfm=valid_tfms)

model = MantisEfficientDet(num_classes=len(datasets.pets.CLASSES), img_size=img_size)
len(datasets.pets.CLASSES)

train_dl = model.dataloader(train_ds, batch_size=16, shuffle=True)
valid_dl = model.valid_dataloader(valid_ds, batch_size=16)


class EfficientDetCallback(fastai.Callback):
    def begin_batch(self):
        assert len(self.xb) == len(self.yb) == 1, "Only works for single input-output"
        self.learn.xb = (self.xb[0], self.yb[0])
        self.learn.yb = ()

    def after_pred(self):
        self.learn.yb = [self.learn.xb[1]]
        self.learn.xb = [self.learn.xb[0]]


cbs = [EfficientDetCallback()]

learn = adapted_fastai_learner(
    dls=[train_dl, valid_dl], model=model, cbs=cbs, loss_func=model.loss
)

learn.fit_one_cycle(1, 1e-3)
