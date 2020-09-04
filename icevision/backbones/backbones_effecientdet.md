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
