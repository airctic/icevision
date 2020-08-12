"""
How to train a voc compatible dataset.
"""

# Installing Mantisshrimp
# !pip install git+git://github.com/airctic/mantisshrimp.git#egg=mantisshrimp[all] --upgrade

# Clone Dataset Repo
# !git clone https://github.com/datitran/raccoon_dataset

# Imports
from mantisshrimp.all import *

# WARNING: Make sure you have already cloned the raccoon dataset using the command shown here above
# Set images and annotations directories
data_dir = Path("raccoon_dataset")
images_dir = data_dir / "images"
annotations_dir = data_dir / "annotations"

# Define class_map
class_map = ClassMap(["raccoon"])

# Parser: Use mantisshrimp pre-defined VOC parser
parser = parsers.voc(
    annotations_dir=annotations_dir, images_dir=images_dir, class_map=class_map
)

# train and validation records
data_splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(data_splitter)
show_records(train_records[:3], ncols=3, class_map=class_map)

# Datasets
# Transforms
presize = 512
size = 384
train_tfms = tfms.A.Adapter(
    [*tfms.A.aug_tfms(size=size, presize=presize), tfms.A.Normalize()]
)
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])

# Train and Validation Dataset Objects
train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

samples = [train_ds[0] for _ in range(3)]
show_samples(samples, ncols=3, class_map=class_map, denormalize_fn=denormalize_imagenet)

# EffecientDet Specific Part

# DataLoaders
train_dl = efficientdet.train_dataloader(
    train_ds, batch_size=16, num_workers=4, shuffle=True
)
valid_dl = efficientdet.valid_dataloader(
    valid_ds, batch_size=16, num_workers=4, shuffle=False
)

# Grab the first batch
batch, samples = first(train_dl)
show_samples(
    samples[:6], class_map=class_map, ncols=3, denormalize_fn=denormalize_imagenet
)

# Model
model = efficientdet.model(
    model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size
)

# Fastai Learner
metrics = [COCOMetric()]
learn = efficientdet.fastai.learner(
    dls=[train_dl, valid_dl], model=model, metrics=metrics
)

# Fastai Training
# Learning Rate Finder
learn.freeze()
learn.lr_find()

# Fine tune: 2 Phases
# Phase 1: Train the head for 10 epochs while freezing the body
# Phase 2: Train both the body and the head during 50 epochs
learn.fine_tune(50, 1e-2, freeze_epochs=10)

# Inference
# DataLoader
infer_dl = efficientdet.infer_dataloader(valid_ds, batch_size=16)
# Predict
samples, preds = efficientdet.predict_dl(model, infer_dl)

# Show samples
imgs = [sample["img"] for sample in samples]
show_preds(
    imgs=imgs[:6],
    preds=preds[:6],
    class_map=class_map,
    denormalize_fn=denormalize_imagenet,
    ncols=3,
)
