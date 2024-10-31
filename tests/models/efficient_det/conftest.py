import pytest


# TODO: Hacky approach
@pytest.fixture
def fridge_efficientdet_records(fridge_ds):
    for i, record in enumerate(fridge_ds[0].records):
        if record.filepath.stem == "10":
            return [fridge_ds[0][i]]
