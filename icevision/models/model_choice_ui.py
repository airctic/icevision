__all__ = ["display_model_choice_ui", "lib_info", "get_model_info"]


from icevision import models
import ipywidgets as widgets
from IPython.display import display


lib_info = {
    "lib_type": None,
    "model_type": None,
    "backbone_type": None,
    "backbone": None,
    "model_list": [],
    "backbone_list": [],
}


def reset_lib_info():
    lib_info.update({"model_list": []})
    lib_info.update({"backbone_list": []})
    lib_info.update({"lib_type": None})
    lib_info.update({"model_type": None})
    lib_info.update({"backbone_type": None})
    lib_info.update({"backbone": None})


def get_model_info():
    model_type = lib_info["model_type"]
    backbone = lib_info["backbone"]

    return model_type, backbone


# Creating dropdown widgets
libraries_available = widgets.Dropdown(
    options=["MMDetection", "Ross Wightman", "Torchvision"],
    # options=[models.ross, models.torchvision],
    description="Libraries",
    disabled=False,
)

models_available = widgets.Dropdown(
    options=[""],
    description="Models",
    disabled=False,
)

backbones_available = widgets.Dropdown(
    options=[""],
    description="Backbones",
    disabled=False,
)


def display_model_choice_ui():
    # Observe dropdown widget changes
    libraries_available.observe(od_library_change, names="value")
    models_available.observe(od_model_change, names="value")
    backbones_available.observe(od_backbone_change, names="value")

    # display dropdown widgets
    display(libraries_available)
    display(models_available)
    display(backbones_available)


def od_library_change(change):
    reset_lib_info()
    lib_name = change.new

    if lib_name == "Torchvision":
        lib_type = models.torchvision
        model_list = ["faster_rcnn", "retinanet", "mask_rcnn", "keypoint_rcnn"]

    if lib_name == "MMDetection":
        lib_type = models.mmdet.models
        model_list = ["sparse_rcnn", "retinanet", "mask_rcnn"]

    if lib_name == "Ross Wightman":
        lib_type = models.ross
        model_list = ["efficientdet"]

    lib_info.update({"lib_type": lib_type})
    lib_info.update({"model_list": model_list})

    models_available.options = model_list


def od_model_change(change):
    model_name = change.new
    lib_type = lib_info["lib_type"]
    model_type = getattr(lib_type, model_name)
    backbone_type = getattr(model_type, "backbones")
    backbone_list = [item for item in dir(model_type.backbones) if "__" not in item]

    lib_info.update({"model_type": model_type})
    lib_info.update({"backbone_type": backbone_type})
    lib_info.update({"backbone_list": backbone_list})
    backbones_available.options = backbone_list


def od_backbone_change(change):
    backbone_name = change.new
    model_type = lib_info["model_type"]
    backbone_type = lib_info["backbone_type"]

    backbone = getattr(backbone_type, backbone_name)

    lib_info.update({"backbone": backbone})
