import pytest
from icevision.all import *


@pytest.fixture()
def classes():
    return ["a", "b", "c"]


def test_class_map_no_background(classes):
    class_map = ClassMap(classes, background=None)
    assert class_map.get_id(0) == "a"
    assert class_map.get_name("a") == 0


def test_class_map_first_background(classes):
    class_map = ClassMap(classes, background=0)
    assert class_map.get_id(0) == "background"
    assert class_map.get_id(1) == "a"
    assert class_map.get_name("a") == 1


def test_class_map_last_background(classes):
    class_map = ClassMap(classes, background=-1)
    assert class_map.get_id(-1) == "background"
    assert class_map.get_id(0) == "a"
    assert class_map.get_name("a") == 0


def test_class_map_safety(classes):
    class_map = ClassMap(classes, background=None)
    classes.insert(0, "x")
    assert class_map.get_id(0) == "a"
