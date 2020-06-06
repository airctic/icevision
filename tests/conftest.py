import pytest
from mantisshrimp import *


@pytest.fixture(scope="module")
def records():
    parser = test_utils.sample_combined_parser()
    return parser.parse()[0]


@pytest.fixture(scope="module")
def record(records):
    return records[2].copy()


@pytest.fixture(scope="module")
def data_sample(record):
    return default_prepare_record(record)
