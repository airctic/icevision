import os
import importlib
from typing import types
from typing import Optional, Union, List
import pathlib

from icevision import models


MODEL_CREATION_FILEPATH = pathlib.Path(__file__).parent.resolve()
BASE_PATH = os.path.join(MODEL_CREATION_FILEPATH, "../models")


def get_module_element_form_module(
    module: types.ModuleType, *args: str
) -> Union[types.ModuleType, callable, object]:
    """Loads a submodule for a given module. Args are the elements to navigate the submodules, the order of the args needs to be the order the submodules appear in (sub)module.

    Parameters
    ----------
    module: types.ModuleType
        Module to load the element from
    args: str
        Strings with the names of the modules/functions/objects in order to retrieve the element

    Example
    -------
    >>> import icevision as iv
    >>> # load the efficientdet model type inside the ross module
    >>> model_type = get_submodule_form_module(iv.models, 'ross', 'efficientdet')

    Returns
    -------
    element_to_return: Union[Module, function, object]
        element that corrosponds to the last element in the args
    """
    element_to_return = module
    for arg in args:
        element_to_return = getattr(element_to_return, arg)
    return element_to_return


def get_backend_libs(base_path: str = BASE_PATH) -> List[str]:
    """Returns all backend_libs available in icevision

    Parameters
    ----------
    base_path: str
        Path from where to read out the available models from

    Returns
    -------
    backend_libs: List[str]
        List of available backend libs
    """
    # FIXME remove the active_backend_libs when the lib restructure is done
    active_backend_libs = ["ross", "torchvision", "fastai", "ultralytics"]
    backend_libs = [
        i
        for i in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, i))
        and not i.startswith("__")
        and i in active_backend_libs
    ]
    return backend_libs


def get_model_types_for_backend_lib(
    backend_lib_name: str, base_path: str = BASE_PATH
) -> List[str]:
    """Returns all available model_types for a given backend lib

    Parameters
    ----------
    backend_lib_name: str
        Name of the backend_lib to get the model types of
    base_path: str
        Path from where to read out the available models from

    Returns
    -------
    model_types: List[str]
        List of available model types
    """
    model_types = [
        i
        for i in os.listdir(os.path.join(base_path, backend_lib_name))
        if os.path.isdir(os.path.join(base_path, backend_lib_name, i))
        and not i.startswith("__")
    ]
    return model_types


def get_backbone_names(
    backend_lib_name: str, model_type_name: str, base_path: str = BASE_PATH
) -> List[str]:
    """Returns all available backbones for a given backend lib

    Parameters
    ----------
    backend_lib_name: str
        Name of the backend_lib to get the backbones for
    model_type_name: str
        Name of the model type to get the backbones for
    base_path: str
        Path from where to read out the available models from

    Returns
    -------
    backbones: List[str]
        List of available backbones
    """
    print(backend_lib_name)
    print(model_type_name)
    backbone_files = [
        i
        for i in os.listdir(
            os.path.join(base_path, backend_lib_name, model_type_name, "backbones")
        )
        if os.path.join(
            base_path, backend_lib_name, model_type_name, "backbones", i
        ).endswith(".py")
        and not i.startswith("__")
    ]
    backbone_list = []
    for backbone_file in backbone_files:
        backbones = importlib.import_module(
            "icevision.models."
            + ".".join([backend_lib_name, model_type_name])
            + ".backbones."
            + backbone_file.replace(".py", "")
        )
        backbone_list += getattr(backbones, "__all__", [])
    return backbone_list


def load_model_components(
    backend_lib_name: str,
    model_type_name: str,
    backbone_name: str,
    model_base_path: str = BASE_PATH,
):
    """Build a model form given backend lib, model_type and backbone.

    Parameters
    ----------
    backend_lib_name: str
        Name of the backend_lib to use
    model_type_name: str
        Name of the model type to use
    backbone_name: str
        Name of the backbone to use

    Returns
    -------
    model_type: Module
        model_type to be used to create the dataloaders and so on
    backbone: object
        backbone for the model
    """
    # load the backend lib
    try:
        backend_lib = get_module_element_form_module(models, backend_lib_name)
    except AttributeError:
        raise AttributeError(
            f"Backend lib: {backend_lib_name} not found. Possible options are: {get_backend_libs(model_base_path)}"
        )
    # load the model_type
    try:
        model_type = get_module_element_form_module(
            models, backend_lib_name, model_type_name
        )
    except AttributeError:
        raise AttributeError(
            f"Model type: {model_type_name} not found. Possible options are: {get_model_types_for_backend_lib(model_base_path, backend_lib_name)}"
        )
    # load backbone
    try:
        backbone_module = get_module_element_form_module(
            models, backend_lib_name, model_type_name, "backbones"
        )
        backbone = getattr(backbone_module, backbone_name)
    except AttributeError:
        raise AttributeError(
            f"Backbone: {backbone_name} not found. Possible options are: {get_backbone_names(backend_lib_name, model_type_name)}"
        )

    return model_type, backbone


def build_model(
    backend_lib_name: str,
    model_type_name: str,
    backbone_name: str,
    num_classes: int,
    backbone_config: Optional[dict] = None,
    model_config: Optional[dict] = None,
):
    """Build a model form given backend lib, model_type and backbone.

    Parameters
    ----------
    backend_lib_name: str
        Name of the backend_lib to use
    model_type_name: str
        Name of the model type to use
    backbone_name: str
        Name of the backbone to use
    num_classes: int
        Number of classes to initialize the model for
    backbone_config: Optional[dict]
        Configuration for the backbone. If None the backbone_config will be set to {"pretrained": True}
    model_config: Optional[dict]
        Configuration for the model


    Returns
    -------
    model_type: model_type to be used to create the dataloaders and so on
    model: model to train
    """
    if backbone_config is None:
        backbone_config = {"pretrained": True}
    if model_config is None:
        model_config = {}
    model_type, backbone = load_model_components(
        backend_lib_name, model_type_name, backbone_name
    )
    model = model_type.model(
        backbone=backbone(**backbone_config), num_classes=num_classes, **model_config
    )
    return model_type, model
