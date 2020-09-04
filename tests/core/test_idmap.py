import pytest
from icevision.all import *


@pytest.fixture
def id_map():
    id_map = IDMap(["file1", "file2"])
    id_map.get_name("file3")
    return id_map


def test_id_map(id_map):
    assert id_map.get_ids() == [0, 1, 2]
    assert id_map.get_names() == ["file1", "file2", "file3"]

    assert id_map.get_id(0) == "file1"
    assert id_map.get_id(2) == "file3"
    assert id_map.get_name("file2") == 1
    assert id_map.get_name("file4") == 3


def test_id_map_filter_ids(id_map):
    filtered_id_map = id_map.filter_ids([0, 2])

    assert filtered_id_map.get_ids() == [0, 2]
    assert filtered_id_map.get_names() == ["file1", "file3"]
