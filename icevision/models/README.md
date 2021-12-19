# Models

[**Source**](https://github.com/airctic/icevision/tree/master/icevision/models)

IceVision offers a large number of models by supporting the following Object Detection Libraries:

* [Torchvision](https://github.com/airctic/icevision/tree/master/icevision/models/torchvision)
* [MMDetection](https://github.com/airctic/icevision/tree/master/icevision/models/mmdet)
* Ross Wightman's [EfficientDet](https://github.com/airctic/icevision/tree/master/icevision/models/ross)

You will enjoy using our unified API while having access to a large repertoire of SOTA models. Switching models is as easy as 
changing one word. There is no need to be familiar with all the quirks that new models and implementations introduce. 

## Creating a model

In order to create a model, we need to:

* Choose one of the **libraries** supported by IceVision
* Choose one of the **models** supported by the library
* Choose one of the **backbones** corresponding to a chosen model

You can access any supported models by following the IceVision unified API, use code completion to explore the available models for each library.

Selecting a model only takes two simple lines of code. Check out the following examples illustrating some of the models libraries we support:

### MMDetection
```
model_type = models.mmdet.retinanet
backbone = model_type.backbones.resnet50_fpn_1x(pretrained=True)
# Instantiate the mdoel
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map))

```

### Torchvision
```
model_type = models.torchvision.retinanet
backbone = model_type.backbones.resnet50_fpn
# Instantiate the mdoel
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map))
```

### EfficientDet
```
model_type = models.ross.efficientdet
backbone = model_type.backbones.tf_lite0
# The efficientdet model requires an img_size parameter
# Instantiate the mdoel
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map), img_size=img_size)
```

### YOLOv5
```
model_type = models.ultralytics.yolov5
backbone = model_type.backbones.small
# The yolov5 model requires an img_size parameter
# Instantiate the mdoel
model = model_type.model(backbone=backbone(pretrained=True), num_classes=len(parser.class_map), img_size=img_size)
```

As pretrained models are used by default, we typically leave this out.
