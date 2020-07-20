import pytest


@pytest.fixture()
def coco_annotations_file(samples_source):
    return samples_source / "annotations.json"
