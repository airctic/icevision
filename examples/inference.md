# How to use the inference API

The inference API is unified one. It is independent from both [Fastai](https://github.com/fastai/fastai) or [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning)


```python
# Installing IceVision
# !pip install icevision[all] icedata

# Imports
from icevision.all import *
import icedata

# Maps IDs to class names. `print(class_map)` for all available classes
class_map = icedata.pets.class_map()

# Try experimenting with new images, be sure to take one of the breeds from `class_map`
IMAGE_URL = "https://petcaramelo.com/wp-content/uploads/2018/06/beagle-cachorro.jpg"
IMG_PATH = "tmp.jpg"
# Model trained in `Tutorials->Getting Started`
WEIGHTS_URL = "https://github.com/airctic/model_zoo/releases/download/pets_faster_resnet50fpn/pets_faster_resnetfpn50.zip"

# Download and open image, optionally show it
download_url(IMAGE_URL, IMG_PATH)
img = open_img(IMG_PATH)
show_img(img, show=True)

# The model was trained with normalized images, it's necessary to do the same in inference
tfms = tfms.A.Adapter([tfms.A.Normalize()])

# Whenever you have images in memory (numpy arrays) you can use `Dataset.from_images`
infer_ds = Dataset.from_images([img], tfms)

# Create the same model used in training and load the weights
# `map_location` will put the model on cpu, optionally move to gpu if necessary
model = faster_rcnn.model(num_classes=len(class_map))
state_dict = torch.hub.load_state_dict_from_url(
    WEIGHTS_URL, map_location=torch.device("cpu")
)
model.load_state_dict(state_dict)

# For any model, the prediction steps are always the same
# First call `build_infer_batch` and then `predict`
batch, samples = faster_rcnn.build_infer_batch(infer_ds)
preds = faster_rcnn.predict(model=model, batch=batch)

# If instead you want to predict in smaller batches, use `infer_dataloader`
infer_dl = faster_rcnn.infer_dl(infer_ds, batch_size=1)
samples, preds = faster_rcnn.predict_dl(model=model, infer_dl=infer_dl)

# Show preds by grabbing the images from `samples`
imgs = [sample["img"] for sample in samples]
show_preds(
    imgs=imgs,
    preds=preds,
    class_map=class_map,
    denormalize_fn=denormalize_imagenet,
    show=True,
)
```