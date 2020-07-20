import pytest


@pytest.fixture()
def fridge_data_dir(samples_source):
    return samples_source / "fridge"
