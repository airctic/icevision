import pytest
from icevision.all import *


def test_multilabel_classification_set_labels():
    class_map = ClassMap(classes=["puppy", "kitten"], background=None)
    comp = MultiLabelClassificationLabelsRecordComponent()
    comp.set_class_map(class_map)
    comp.set_labels(["puppy"])

    assert comp.is_one_hot_encoded
    assert isinstance(comp.labels, np.ndarray)
    assert (comp.labels == np.array([1, 0])).all()


def test_multilabel_classification_set_labels_elaborate():
    class_map = ClassMap(classes=["puppy", "kitten", "wolf", "cow"], background=None)
    comp = MultiLabelClassificationLabelsRecordComponent()
    comp.set_class_map(class_map)
    comp.set_labels(["kitten", "cow"])

    assert comp.is_one_hot_encoded
    assert isinstance(comp.labels, np.ndarray)
    assert (comp.labels == np.array([0, 1, 0, 1])).all()


def test_multilabel_classification_add_labels():
    class_map = ClassMap(classes=["puppy", "kitten"], background=None)
    comp = MultiLabelClassificationLabelsRecordComponent()
    comp.set_class_map(class_map)

    with pytest.raises(NotImplementedError):
        comp.add_labels(["puppy"])
