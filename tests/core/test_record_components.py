import pytest
from icevision.all import *
from PIL import Image


@pytest.fixture
def dummy_class_map():
    return ClassMap(["dummy-1", "dummy-2"], background=None)


@pytest.fixture
def dummy_class_map_elaborate():
    return ClassMap(["dummy-1", "dummy-2", "dummy-3", "dummy-4"], background=None)


def test_classification_multilabel(dummy_class_map):
    rec = BaseRecord([ClassificationLabelsRecordComponent(is_multilabel=True)])
    rec.classification.set_class_map(dummy_class_map)
    rec.classification.set_labels_by_id([0, 1])

    assert rec.classification.label_ids == [0, 1]
    assert (rec.classification.one_hot_encoded() == np.array([1, 1])).all()


@pytest.mark.parametrize(
    "label_ids",
    [
        ([0, 1]),
        ([0]),
    ],
)
def test_classification_single_label(dummy_class_map, label_ids):
    rec = BaseRecord([ClassificationLabelsRecordComponent(is_multilabel=False)])
    rec.classification.set_class_map(dummy_class_map)
    rec.classification.set_labels_by_id(label_ids)

    if len(label_ids) > 1:
        # label_ids == [0, 1]
        # Setting two labels when `is_multilabel=False` raises an error
        with pytest.raises(AutofixAbort):
            rec.classification._autofix()
    else:
        # label_ids == [0]
        # Only one label must be assigned
        assert all(rec.classification._autofix().values())
        assert rec.classification.one_hot_encoded().sum() == 1


@pytest.mark.parametrize(
    "label_ids",
    [
        ([0, 1, 2]),
        ([0, 1]),
        ([0]),
    ],
)
def test_one_hot_encodings(dummy_class_map_elaborate, label_ids):
    rec = BaseRecord([ClassificationLabelsRecordComponent(is_multilabel=True)])
    rec.classification.set_class_map(dummy_class_map_elaborate)
    rec.classification.set_labels_by_id(label_ids)

    assert all(rec.classification._autofix().values())

    # Ensure we have the correct no. of labels and that they are indeed
    # one-hot encoded
    one_hot_values = rec.classification.one_hot_encoded()
    assert one_hot_values.sum() == len(label_ids)
    assert np.unique(one_hot_values).tolist() == [0, 1]


def test_record_repr_img_is_none():
    test_img_record_component = core.record_components.ImageRecordComponent()
    assert test_img_record_component._repr() == ["Img: None"]


def test_record_repr_img_is_array():
    test_img_record_component = core.record_components.ImageRecordComponent()
    test_img_record_component.img = np.ones([4, 4, 3], dtype=np.uint8)
    assert test_img_record_component._repr() == ["Img: 4x4x3 <np.ndarray> Image"]


def test_record_repr_img_is_array():
    test_img_record_component = core.record_components.ImageRecordComponent()
    test_img_record_component.img = PIL.Image.fromarray(
        np.ones([4, 4, 3], dtype=np.uint8)
    )
    assert test_img_record_component._repr() == ["Img: 4x4 <PIL.Image; mode='RGB'>"]


def test_RecordComponent_as_dict_returns_empty():
    record_component = core.record_components.RecordComponent()
    assert record_component.as_dict() == {}


def test_RecordComponent_aggregate_objects_returns_empty():
    record_component = core.record_components.RecordComponent()
    assert record_component._aggregate_objects() == {}


def test_RecordComponent_repr_returns_empty():
    record_component = core.record_components.RecordComponent()
    assert record_component._repr() == []


def test_RecordComponent_builder_template_returns_empty():
    record_component = core.record_components.RecordComponent()
    assert record_component._builder_template() == []


def test_ClassMapRecordComponent_repr():
    class_map_record_component = core.record_components.ClassMapRecordComponent(
        "object_detection"
    )
    class_map_record_component.class_map = ClassMap(["test"])
    assert class_map_record_component._repr() == [
        "Class Map: <ClassMap: {'background': 0, 'test': 1}>"
    ]
