from mantisshrimp import *
from mantisshrimp.imports import *
from mantisshrimp.models import efficientdet
import albumentations as A


IMG_SIZE = 512
class_map = datasets.fridge.class_map()
data_dir = datasets.fridge.load()
parser = datasets.fridge.parser(data_dir, class_map)

data_splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(data_splitter)

imagenet_mean, imagenet_std = IMAGENET_STATS

valid_tfms = AlbumentationTransforms(
    [A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(mean=imagenet_mean, std=imagenet_std),]
)

train_tfms = AlbumentationTransforms(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        # A.RandomSizedBBoxSafeCrop(320, 320, p=0.3),
        A.HorizontalFlip(),
        A.ShiftScaleRotate(rotate_limit=20),
        A.RGBShift(always_apply=True),
        A.RandomBrightnessContrast(),
        A.Blur(blur_limit=(1, 3)),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

train_dl = efficientdet.train_dataloader(
    train_ds, batch_size=16, num_workers=4, shuffle=True
)
valid_dl = efficientdet.valid_dataloader(valid_ds, batch_size=16, num_workers=4)

model = efficientdet.model(
    "tf_efficientdet_lite0", num_classes=len(class_map), img_size=IMG_SIZE
)

metrics = [COCOMetric(print_summary=True)]
learn = efficientdet.fastai.learner(
    dls=[train_dl, valid_dl], model=model, metrics=metrics
)

learn.lr_find()
learn.fine_tune(50, 1e-2, freeze_epochs=20)

# Inference
WEIGHTS_URL = "https://mantisshrimp-models.s3.us-east-2.amazonaws.com/fridge_tf_efficientdet_lite0.zip"
model = efficientdet.model("tf_efficientdet_lite0", num_classes=5, img_size=512)

state_dict = torch.hub.load_state_dict_from_url(
    WEIGHTS_URL, map_location=torch.device("cpu")
)
model.load_state_dict(state_dict)
model.cuda()


samples = [valid_ds[i] for i in range(3)]
batch, samples = efficientdet.build_infer_batch(samples)

preds = efficientdet.predict(model, batch)

imgs = [sample["img"] for sample in samples]
show_preds(imgs=imgs, preds=preds)
