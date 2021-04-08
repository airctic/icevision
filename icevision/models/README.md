# Models

[**Source**](https://github.com/airctic/icevision/tree/master/icevision/models)

IceVision offers a large number of models by supporting the following Object Detection Libraries:

* [Torchvision](https://github.com/airctic/icevision/tree/master/icevision/models/torchvision)
* [MMDetection](https://github.com/airctic/icevision/tree/master/icevision/models/mmdet)
* Ross Wightman's [EfficicientDet](https://github.com/airctic/icevision/tree/master/icevision/models/ross)

You will enjoy using our unified API while having access to a large repertoire of SOTA models. Switching models is as easy as 
changing one word. There is no need to be familiar with all the quirks that new models and implementations introduce. 

Thanks to IceVision Unified API, model are created in a very similar way.

```python
model = model_type.model(backbone=backbone(), num_classes=len(parser.class_map))
```
