from icevision.core.components.component_registry import *

__all__ = [
    "ImageidComponent",
    "ClassMapComponent",
    "SizeComponent",
    "ImageArrayComponent",
    "FilepathComponent",
    "BBoxComponent",
    "LabelComponent",
    "MaskComponent",
    "AreaComponent",
    "IsCrowdComponent",
    "KeyPointComponent",
]

ImageidComponent = component_registry.new_component_registry("imageid")
ClassMapComponent = component_registry.new_component_registry("classmap")
SizeComponent = component_registry.new_component_registry("size")
ImageArrayComponent = component_registry.new_component_registry("image_array")
FilepathComponent = component_registry.new_component_registry("filepath")
BBoxComponent = component_registry.new_component_registry("bbox")
LabelComponent = component_registry.new_component_registry("label")
MaskComponent = component_registry.new_component_registry("mask")
AreaComponent = component_registry.new_component_registry("area")
IsCrowdComponent = component_registry.new_component_registry("iscrowd")
KeyPointComponent = component_registry.new_component_registry("keypoint")
