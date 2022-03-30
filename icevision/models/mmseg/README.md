# Introduction

This is the integration of `mmsegmentation`, adding support for a wide range of semantic segmentation models and architectures.

# How the integration works

The integration is very similar to that of `mmdet` (as both libraries share much of their structure).

There are multiple layers of integrations, which are described here to facilitate future developments. 

## Callbacks

The [`MMSegmentationCallback`](icevision/models/mmseg/fastai/callbacks.py) is responsible for making the `mmsegmentation` models compatible with the fastai training loop. It ensures that:
- The `after_create` hook is used to wrap the model in a way that `mmsegmentation`'s conventions are translated to fit with fastai's model operations
- The `after_loss` hook is used to convert model predictions into the `IceVision` standardized record format, which allows predictions to be manipulated and visualized using the library's standard features

## Dataloaders

Input / output handling in `mmsegmentation` is significantly different to the way that is conventionally done in fastai or PyTorch. Custom dataloaders are used to make sure that batches are created in the format expected by `mmsegmentation` (see the [doc for more information](https://mmsegmentation.readthedocs.io/en/latest/api.html#mmseg.models.segmentors.BaseSegmentor.forward_test))

# FAQ

**What architectures are currently supported?** 
* `DeepLabV3`
* `DeepLabV3+`
* `SegFormer`

**I am encountering a *CUDA error: an illegal memory access was encountered* error** 
It is likely that the number of classes hasn't been set properly