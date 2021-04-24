import pytest
from icevision.all import *


@pytest.fixture
def dummy_class_map():
    return ClassMap(["dummy-1", "dummy-2"], background=None)


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
        with pytest.raises(ValueError):
            rec.classification._load()
    else:
        # Only one label must be assigned
        rec.classification._load()
        assert rec.classification.one_hot_encoded().sum() == 1
