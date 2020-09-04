from icevision.all import *


def test_coco_class_map(coco_annotations_file):
    class_map = datasets.coco.class_map(coco_annotations_file, background=0)

    assert class_map.get_id(0) == "background"
    assert class_map.get_id(2) == "bicycle"
    assert class_map.get_name("car") == 3
