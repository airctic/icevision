from icevision.all import *


def test_fridge_class_map():
    class_map = datasets.fridge.class_map()

    assert len(class_map) == 5
    assert class_map.get_id(0) == "background"
    assert class_map.get_id(-1) == "water_bottle"


def test_fridge_parser(fridge_data_dir):
    class_map = datasets.fridge.class_map()
    parser = datasets.fridge.parser(fridge_data_dir, class_map=class_map)

    records = parser.parse()[0]
    assert len(records) == 1
    record = records[0]

    expected = {"imageid", "labels", "bboxes", "filepath", "height", "width"}
    assert set(record.keys()) == expected

    assert record["filepath"].name == "10.jpg"
    assert record["imageid"] == 0
    assert record["labels"] == [2, 3]
    assert record["height"] == 666
    assert record["width"] == 499

    expected = [BBox.from_xyxy(86, 102, 216, 439), BBox.from_xyxy(150, 377, 445, 490)]
    assert record["bboxes"] == expected
