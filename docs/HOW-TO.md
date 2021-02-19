## Where can I get some help?
- If you find a bug, or you would like to suggest some new features, please file an issue [here](https://github.com/airctic/icevision/issues)

- If you need any assistance during your learning journey, feel free to join our [forum](https://discord.gg/JDBeZYK).


## How to install icevision?
To install the IceVision package as well as all its dependencies, choose one of the 2 options:

Installing the IceVision lastest version

```bash
pip install git+git://github.com/airctic/icevision.git#egg=icevision[all] --upgrade
```

Install the IceVision lastest version from Pypi repository:
```bash
pip install icevision[all]
```

For more options, and more in-depth explanation on how to install IceVision, please check out our [Installation Guide](https://airctic.github.io/icevision/install/
) 

## How to create an EffecientDet Model?

**tf_efficientdet_lite0** Example: [Source Code](https://airctic.github.io/icevision/examples/efficientdet_pets_exp/)

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

For more information checkout the [EffecientDet Model](https://airctic.github.io/icevision/model_efficientdet/) as well as the [EffecientDet Backbone](https://airctic.com/backbones_effecientdet/) documents.


## How to create a Faster RCNN Model?
**fasterrcnn_resnet50_fpn** Example: [Source Code](https://airctic.github.io/icevision/examples/backbones_faster_rcnn/)

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

For more information checkout the [Faster RCNN Model](https://airctic.github.io/icevision/model_faster_rcnn/) as well as the [Faster RCNN Backbone](https://airctic.com/backbones_faster_mask_rcnn/#faster-rcnn-backbones-examples) documents/


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

For more information checkout the [Faster RCNN Model](https://airctic.github.io/icevision/model_faster_rcnn/) as well as the [Faster RCNN Backbone](https://airctic.com/backbones_faster_mask_rcnn/#mask-rcnn-backbones-examples) documents.

## How to use EffecientDet Backbones?
EffecientDet backbones are passed as string argument to the effecientdet model function:

``` python hl_lines="2"
model = efficientdet.model(
    model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size
)
```
For more information checkout the [EffecientDet Backbone](https://airctic.com/backbones_effecientdet/) document.

## How to use Faster RCNN Backbones?
Faster RCNN backbones are passed a model object argument to the Faster RCNN model function:

``` python hl_lines="1 3"
backbone = backbones.resnet_fpn.resnet18(pretrained=True)
model = faster_rcnn.model(
    backbone=backbone, num_classes=len(class_map)
)
```
For more information checkout the [Faster RCNN Backbone](https://airctic.com/backbones_faster_mask_rcnn/#faster-rcnn-backbones-examples) document.

## How to use Mask RCNN Backbones?
Mask RCNN backbones are passed a model object argument to the Mask RCNN model function:

``` python hl_lines="1 3"
backbone = backbones.resnet_fpn.resnet34(pretrained=True)
model = mask_rcnn.model(
    backbone=backbone, num_classes=len(class_map)
)
```
For more information checkout the [Faster RCNN Backbone](https://airctic.com/backbones_faster_mask_rcnn/#mask-rcnn-backbones-examples) document.

## How to predict (infer) a single image?
This is a quick example using the PETS dataset:

```python hl_lines="14-16 22"
# Imports
from icevision.all import *

# Maps from IDs to class names. `print(class_map)` for all available classes
class_map = datasets.pets.class_map()

# Try experimenting with new images, be sure to take one of the breeds from `class_map`
IMAGE_URL = "https://petcaramelo.com/wp-content/uploads/2018/06/beagle-cachorro.jpg"
IMG_PATH = "tmp.jpg"
# Model trained on `Tutorials->Getting Started`
WEIGHTS_URL = "https://github.com/airctic/model_zoo/releases/download/m3/pets_faster_resnetfpn50.zip"

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

## How to save trained weights in Google Colab?
In the following example, we show how to save trained weight using an EffecientDet model. The latter can be replaced by any model supported by IceVision

Check out the [Quick Start Notebook](https://airctic.com/quickstart/) to get familiar with all the steps from the training a dataset to saving the trained weights. 

```python
# Model
model = efficientdet.model(
    model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size
)
# Train the model using either Fastai Learner of Pytorch-Lightning Trainer

## Saving a Model on Google Drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = Path('/content/gdrive/My Drive/')

torch.save(model.state_dict(), root_dir/'icevision/models/fridge/fridge_tf_efficientdet_lite0.pth')
```

## How to load pretrained weights?
In this example, we show how to create a Faster RCNN model, and load pretrained weight that were previously obtained during the training of the PETS dataset as shown in the [Getting Started Notebook](https://airctic.github.io/icevision/getting_started/)

```python
# Maps IDs to class names.
class_map = datasets.pets.class_map()

# Model trained in `Tutorials->Getting Started`
WEIGHTS_URL = "https://github.com/airctic/model_zoo/releases/download/m3/pets_faster_resnetfpn50.zip"

# Create the same model used in training and load the weights
# `map_location` will put the model on cpu, optionally move to gpu if necessary
model = faster_rcnn.model(num_classes=len(class_map))
state_dict = torch.hub.load_state_dict_from_url(
    WEIGHTS_URL, map_location=torch.device("cpu")
)
model.load_state_dict(state_dict)
```

## How to contribute?
We are both a welcoming and an open community. We warmly invite you to join us either as a user or a community contributor. We will be happy to hear from you.

To contribute, please follow the [Contributing Guide](https://airctic.github.io/icevision/contributing/). 
