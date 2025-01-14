import pytest
import icevision.widgets as iw


def test_ModelChoiceUI_init_default():
    model_choice_ui = iw.ModelChoiceUI()
    assert model_choice_ui.task == "object_detection"


def test_ModelChoiceUI_init_object_detection():
    model_choice_ui = iw.ModelChoiceUI("object_detection")
    assert model_choice_ui.task == "object_detection"


def test_ModelChoiceUI_init_mask():
    model_choice_ui = iw.ModelChoiceUI("mask")
    assert model_choice_ui.task == "mask"


def test_ModelChoiceUI_init_keypoints():
    model_choice_ui = iw.ModelChoiceUI("keypoints")
    assert model_choice_ui.task == "keypoints"


def test_ModelChoiceUI_display():
    model_choice_ui = iw.ModelChoiceUI()
    _ = model_choice_ui.display()


@pytest.mark.parametrize(
    "task, libs, models",
    [
        (
            "object_detection",
            ["MMDetection", "Ross Wightman", "Torchvision"],
            {
                "MMDetection": ["retinanet", "faster_rcnn", "fcos", "sparse_rcnn"],
                "Ross Wightman": ["efficientdet"],
                "Torchvision": ["faster_rcnn", "retinanet"],
            },
        ),
        (
            "mask",
            ["MMDetection", "Torchvision"],
            {"MMDetection": ["mask_rcnn"], "Torchvision": ["mask_rcnn"]},
        ),
        ("keypoints", ["Torchvision"], {"Torchvision": ["keypoint_rcnn"]}),
    ],
)
def test_ModelChoiceUI_change_widget_values(task, libs, models):
    model_choice_ui = iw.ModelChoiceUI(task)
    _ = model_choice_ui.display()
    for lib in libs:
        model_choice_ui.libraries_available.value = lib
        for model in models[lib]:
            model_choice_ui.models_available.value = model


def test_ModelChoiceUI_get_model_info():
    model_choice_ui = iw.ModelChoiceUI()
    _ = model_choice_ui.display()
    _ = model_choice_ui.get_model_info()
