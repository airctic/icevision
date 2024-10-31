import pytest
from icevision.all import *


@pytest.fixture
def records():
    def create_record_func():
        return BaseRecord([FilepathRecordComponent()])

    records = RecordCollection(create_record_func)
    for record_id, filepath in zip(
        ["file1", "file2", "file3", "file4"],
        ["/train/0", "/train/1", "/valid/0", "/test/0"],
    ):
        record = records.get_by_record_id(record_id)
        record.set_filepath(filepath)
    return records


def test_single_split_splitter(records):
    data_splitter = SingleSplitSplitter()
    splits = data_splitter(records)
    assert splits == [["file1", "file2", "file3", "file4"]]


def test_random_splitter(records):
    data_splitter = RandomSplitter([0.6, 0.2, 0.2], seed=42)
    splits = data_splitter(records)
    np.testing.assert_equal(splits, [["file2", "file4"], ["file1"], ["file3"]])


def test_fixed_splitter(records):
    presplits = [["file4", "file3"], ["file2"], ["file1"]]

    data_splitter = FixedSplitter(presplits)
    splits = data_splitter(records)
    assert splits == presplits


def test_folder_splitter(records):
    presplits = [{"file1", "file2"}, {"file3"}, {"file4"}]

    keywords = ["train", "valid", "test"]
    data_splitter = FolderSplitter(keywords=keywords)
    splits = data_splitter(records)

    assert splits == presplits


def test_folder_splitter_with_remainder(records):
    presplits = [{"file1", "file2"}, {"file4"}, {"file3"}]

    keywords = ["train", "test"]
    data_splitter = FolderSplitter(keywords=keywords)
    splits = data_splitter(records)

    assert splits == presplits


def test_record_collection_adding(records):
    copy = deepcopy(records)
    for record_id in ["file5", "file6", "file7", "file8"]:
        copy.get_by_record_id(record_id)
    records_sum = copy + records
    assert len(records_sum) == 2 * len(records)


def test_record_collection_slicing(records):
    subset = records[:2]
    assert isinstance(subset, RecordCollection)
