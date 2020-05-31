import pytest
from mantisshrimp import *


@pytest.fixture(scope="module")
def item():
    parser = test_utils.sample_data_parser()
    with np_local_seed(42):
        train_rs, valid_rs = parser.parse(show_pbar=False)
    return Item.from_record(train_rs[0])
