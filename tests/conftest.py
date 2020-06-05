import pytest
from mantisshrimp import *


@pytest.fixture(scope="module")
def records():
    parser = test_utils.sample_combined_parser()
    return parser.parse()[0]


@pytest.fixture(scope="module")
def record(records):
    return records[2]


@pytest.fixture(scope="module")
def data_sample(record):
    data_preparer = DefaultDataPreparer()
    return data_preparer(record)
