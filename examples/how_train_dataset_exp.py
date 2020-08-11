"""
Example showing how to train the PETS dataset, showcasing [fastai2](https://github.com/fastai/fastai2)
"""

# Installing Mantisshrimp
# !pip install git+git://github.com/airctic/mantisshrimp.git#egg=mantisshrimp[all] --upgrade

# Imports
from mantisshrimp.all import *

# Common part to all models

# Loading Data
data_dir = datasets.fridge.load()

# Parser
class_map = datasets.fridge.class_map()
parser = datasets.fridge.parser(data_dir, class_map)
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
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])

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
learn.freeze()
learn.lr_find()

learn.fine_tune(50, 1e-2, freeze_epochs=20)

# Inference
# DataLoader
infer_dl = efficientdet.infer_dataloader(valid_ds, batch_size=8)
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
