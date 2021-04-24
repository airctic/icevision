import pytest
from icevision.all import *


@pytest.fixture()
def classes():
    return ["a", "b", "c"]


def test_class_map_no_background(classes):
    class_map = ClassMap(classes, background=None)
    assert class_map.get_by_id(0) == "a"
    assert class_map.get_by_name("a") == 0


@pytest.mark.parametrize("background", [BACKGROUND, "bg", 0])
def test_class_map_background(classes, background):
    class_map = ClassMap(classes, background=background)
    assert class_map.get_by_id(0) == background
    assert class_map.get_by_name(background) == 0
    assert class_map.get_by_id(1) == "a"
    assert class_map.get_by_name("a") == 1


def test_class_map_safety(classes):
    class_map = ClassMap(classes, background=None)
    classes.insert(0, "x")
    assert class_map.get_by_id(0) == "a"

    with pytest.raises(ValueError):
        class_map.add_name(class_map._id2class[0])
