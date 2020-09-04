# Backbones

EffecientDet, Faster RCNN, and Mask RCNN support different backbones. 

# EffecientDet Backbones

[**Source**](https://github.com/rwightman/efficientdet-pytorch)


## Usage

We use Ross Wightman's implementation which is an accurate port of the official TensorFlow (TF) implementation that accurately preserves the TF training weights

[EfficientDet (PyTorch)](https://github.com/rwightman/efficientdet-pytorch)

Any backbone in the timm model collection that supports feature extraction (features_only arg) can be used as a bacbkone.
We can choose one of the **efficientdet_d0** to **efficientdet_d7** backbones, and **MobileNetv3** classes (which also includes **MNasNet**, **MobileNetV2**, **MixNet** and more)

#### **EffecientDet Backbones Examples**

**tf_efficientdet_lite0** Example: [Source Code](https://airctic.github.io/icevision/examples/efficientdet_pets_exp/)

``` python hl_lines="2"
model = efficientdet.model(
    model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size
)
```

**efficientdet_d0** Example:

``` python hl_lines="2"
model = efficientdet.model(
    model_name="efficientdet_d0", num_classes=len(class_map), img_size=size
)
```

#### Supported Backbones
**EffecientDet Backbones**

- tf_efficientdet_lite0

- efficientdet_d0

- efficientdet_d1

- efficientdet_d2

- efficientdet_d3

- efficientdet_d4

- efficientdet_d5

- efficientdet_d6

- efficientdet_d7

- efficientdet_d7x


**MobileNetv3**

**MNasNet**

**MobileNetV2**

**MixNet**

# Faster RCNN / Mask RCNN Backbones

[**Source**](https://github.com/airctic/icevision/tree/master/icevision/backbones)


## Usage

We use the [torchvision Faster RCNN](https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py)  model, and the [torchvision Mask RCNN](https://github.com/pytorch/vision/blob/master/torchvision/models/detection/mask_rcnn.py) model. 

Both models accept a variety of backbones. In following example, we use the default [fasterrcnn_resnet50_fpn](https://github.com/pytorch/vision/blob/27278ec8887a511bd7d6f1202d50b0da7537fc3d/torchvision/models/detection/faster_rcnn.py#L291) model. We can also choose one of the many [backbones](https://github.com/airctic/icevision/blob/master/icevision/backbones/resnet_fpn.py) listed here below: 

#### **Faster RCNN Backbones Examples**
**fasterrcnn_resnet50_fpn** Example: [Source Code](https://airctic.github.io/icevision/examples/backbones_faster_rcnn/)

**- Using the default argument**
``` python
model = faster_rcnn.model(num_classes=len(class_map))
```

**Using the explicit backbone definition**
``` python hl_lines="1 3"
backbone = backbones.resnet_fpn.resnet50(pretrained=True) # Default
model = faster_rcnn.model(
    backbone=backbone, num_classes=len(class_map)
)
```

**resnet18** Example:

``` python hl_lines="1 3"
backbone = backbones.resnet_fpn.resnet18(pretrained=True)
model = faster_rcnn.model(
    backbone=backbone, num_classes=len(class_map)
)
```

#### **Mask RCNN Backbones Examples**
**fasterrcnn_resnet50_fpn** Example:

**- Using the default argument**
``` python
model = mask_rcnn.model(num_classes=len(class_map))
```

**Using the explicit backbone definition**
``` python hl_lines="1 3"
backbone = backbones.resnet_fpn.resnet50(pretrained=True) # Default
model = mask_rcnn.model(
    backbone=backbone, num_classes=len(class_map)
)
```

**resnet34** Example:

``` python hl_lines="1 3"
backbone = backbones.resnet_fpn.resnet34(pretrained=True)
model = faster_rcnn.model(
    backbone=backbone, num_classes=len(class_map)
)
```

#### Supported Backbones
**FPN backbones**
- resnet18

- resnet34

- resnet50

- resnet101

- resnet152

- resnext50_32x4d

- resnext101_32x8d

- wide_resnet50_2

- wide_resnet101_2

**Resnet backbone**
- resnet18

- resnet34

- resnet50

- resnet101

- resnet152

- resnext101_32x8d

**MobileNet**
- mobilenet

**VGG**

- vgg11

- vgg13

- vgg16

- vgg19

