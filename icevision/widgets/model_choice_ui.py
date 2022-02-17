__all__ = ["ModelChoiceUI"]


from icevision import models
import ipywidgets as widgets
from IPython.display import display


class ModelChoiceUI:
    def __init__(self, task="object_detection"):
        self.task = task
        self.reset_lib_info()

        self.libraries_available = widgets.Dropdown(
            options=[],
            description="Libraries",
            disabled=False,
        )

        self.models_available = widgets.Dropdown(
            options=[""],
            description="Models",
            disabled=False,
        )

        self.backbones_available = widgets.Dropdown(
            options=[""],
            description="Backbones",
            disabled=False,
        )

    # lib_info = {
    #     "lib_type": None,
    #     "model_type": None,
    #     "backbone_type": None,
    #     "backbone": None,
    #     "model_list": [],
    #     "backbone_list": [],
    # }

    def reset_lib_info(self):
        self.lib_type = None
        self.model_type = None
        self.backbone_type = None
        self.model_list = []
        self.backbone_list = []
        self.backbone = None

    def get_model_info(self):
        return self.model_type, self.backbone

    # Creating dropdown widgets
    def populate_libraries(self):
        if self.task == "object_detection":
            libraries_list = ["", "MMDetection", "Ross Wightman", "Torchvision"]
        elif self.task == "mask":
            libraries_list = ["", "MMDetection", "Torchvision"]
        elif self.task == "keypoints":
            libraries_list = ["", "Torchvision"]

        self.libraries_available.options = libraries_list

    def od_library_change(self, change):
        lib_name = change.new

        if self.task == "object_detection":
            if lib_name == "Torchvision":
                lib_type = models.torchvision
                model_list = ["faster_rcnn", "retinanet"]

            if lib_name == "MMDetection":
                lib_type = models.mmdet.models
                model_list = ["retinanet", "faster_rcnn", "fcos", "sparse_rcnn"]

            if lib_name == "Ross Wightman":
                lib_type = models.ross
                model_list = ["efficientdet"]

        elif self.task == "mask":
            if lib_name == "Torchvision":
                lib_type = models.torchvision
                model_list = ["mask_rcnn"]

            if lib_name == "MMDetection":
                lib_type = models.mmdet.models
                model_list = ["mask_rcnn"]

        elif self.task == "keypoints":
            if lib_name == "Torchvision":
                lib_type = models.torchvision
                model_list = ["keypoint_rcnn"]

        self.lib_type = lib_type
        self.model_list = model_list

        self.models_available.options = model_list

    def od_model_change(self, change):
        model_name = change.new
        lib_type = self.lib_type
        model_type = getattr(lib_type, model_name)
        backbone_type = getattr(model_type, "backbones")
        backbone_list = [item for item in dir(model_type.backbones) if "__" not in item]

        self.model_type = model_type
        self.backbone_type = backbone_type
        self.backbone_list = backbone_list
        self.backbones_available.options = backbone_list

    def od_backbone_change(self, change):
        backbone_name = change.new
        model_type = self.model_type
        backbone_type = self.backbone_type

        backbone = getattr(backbone_type, backbone_name)

        self.backbone = backbone

    def display(self):
        self.populate_libraries()
        # Observe dropdown widget changes
        self.libraries_available.observe(self.od_library_change, names="value")
        self.models_available.observe(self.od_model_change, names="value")
        self.backbones_available.observe(self.od_backbone_change, names="value")

        # display dropdown widgets
        display(self.libraries_available)
        display(self.models_available)
        display(self.backbones_available)
