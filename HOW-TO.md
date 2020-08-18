## How to install mantisshrimp?
To install the Mantisshrimp package as well as all its dependencies, choose one of the 2 options:

Installing the Mantishrimp lastest version

```bash
pip install git+git://github.com/airctic/mantisshrimp.git#egg=mantisshrimp[all] --upgrade
```

Install the Mantishrimp lastest version from Pypi repository:
```bash
pip install mantisshrimp[all]
```

For more options, and more in-depth explanation on how to install Mantisshrimp, please check out our [Installation Guide](https://airctic.github.io/mantisshrimp/install/
) 

## How to create an EffecientDet Model?

**tf_efficientdet_lite0** Example: [Source Code](https://airctic.github.io/mantisshrimp/examples/efficientdet_pets_exp/)

``` python
model = efficientdet.model(
    model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size
)
```

**efficientdet_d0** Example:

``` python
model = efficientdet.model(
    model_name="efficientdet_d0", num_classes=len(class_map), img_size=size
)
```

For more information checkout the [EffecientDet Model](https://airctic.github.io/mantisshrimp/model_efficientdet/) as well as the [EffecientDet Backbone](https://airctic.github.io/mantisshrimp/backbones_overview/) documents.


## How to create a Faster RCNN Model?
**fasterrcnn_resnet50_fpn** Example: [Source Code](https://airctic.github.io/mantisshrimp/examples/backbones_faster_rcnn/)

**- Using the default argument**
``` python
model = faster_rcnn.model(num_classes=len(class_map))
```

**Using the explicit backbone definition**
``` python
backbone = backbones.resnet_fpn.resnet50(pretrained=True) # Default
model = faster_rcnn.model(
    backbone=backbone, num_classes=len(class_map)
)
```

For more information checkout the [Faster RCNN Model](https://airctic.github.io/mantisshrimp/model_faster_rcnn/) as well as the [Faster RCNN Backbone](https://airctic.github.io/mantisshrimp/backbones_overview/) documents/


## How to create a Mask RCNN Model?

**- Using the default argument**
``` python
model = mask_rcnn.model(num_classes=len(class_map))
```

**Using the explicit backbone definition**
``` python
backbone = backbones.resnet_fpn.resnet50(pretrained=True) # Default
model = mask_rcnn.model(
    backbone=backbone, num_classes=len(class_map)
)
```

For more information checkout the [Faster RCNN Model](https://airctic.github.io/mantisshrimp/model_faster_rcnn/) as well as the [Faster RCNN Backbone](https://airctic.github.io/mantisshrimp/backbones_overview/) documents.

## How to use EffecientDet Backbones?
EffecientDet backbones are passed as string argument to the effecientdet model function:

``` python hl_lines="2"
model = efficientdet.model(
    model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size
)
```
For more information checkout the [EffecientDet Backbone](https://airctic.github.io/mantisshrimp/backbones_overview/) document.

## How to use Faster RCNN Backbones?
Faster RCNN backbones are passed a model object argument to the Faster RCNN model function:

``` python hl_lines="1 3"
backbone = backbones.resnet_fpn.resnet18(pretrained=True)
model = faster_rcnn.model(
    backbone=backbone, num_classes=len(class_map)
)
```
For more information checkout the [Faster RCNN Backbone](https://airctic.github.io/mantisshrimp/backbones_overview/) document.

## How to use Mask RCNN Backbones?
Mask RCNN backbones are passed a model object argument to the Mask RCNN model function:

``` python hl_lines="1 3"
backbone = backbones.resnet_fpn.resnet34(pretrained=True)
model = mask_rcnn.model(
    backbone=backbone, num_classes=len(class_map)
)
```
For more information checkout the [Faster RCNN Backbone](https://airctic.github.io/mantisshrimp/backbones_overview/) document.

## How to predict (infer) a single image?
This is a quick example using the PETS dataset:

```python hl_lines="14-16 22"
# Imports
from mantisshrimp.all import *

# Maps from IDs to class names. `print(class_map)` for all available classes
class_map = datasets.pets.class_map()

# Try experimenting with new images, be sure to take one of the breeds from `class_map`
IMAGE_URL = "https://petcaramelo.com/wp-content/uploads/2018/06/beagle-cachorro.jpg"
IMG_PATH = "tmp.jpg"
# Model trained on `Tutorials->Getting Started`
WEIGHTS_URL = "https://mantisshrimp-models.s3.us-east-2.amazonaws.com/pets.zip"

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