import pytest
from mantisshrimp.all import *


@pytest.fixture
def idmap():
    return IDMap(["file1", "file2", "file3", "file4"])


def test_single_split_splitter(idmap):
    data_splitter = SingleSplitSplitter()
    splits = data_splitter(idmap)
    assert splits == [[0, 1, 2, 3]]


def test_random_splitter(idmap):
    data_splitter = RandomSplitter([0.6, 0.2, 0.2], seed=42)
    splits = data_splitter(idmap)
    np.testing.assert_equal(splits, [[1, 3], [0], [2]])


def test_fixed_splitter(idmap):
    presplits = [["file4", "file3"], ["file2"], ["file1"]]

    data_splitter = FixedSplitter(presplits)
    splits = data_splitter(idmap=idmap)
    assert splits == [[3, 2], [1], [0]]
