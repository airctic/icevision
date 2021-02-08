from icevision.all import *


def test_dataset_from_images():
    images = np.zeros([2, 4, 4, 3])
    dataset = Dataset.from_images(images)

    assert len(dataset) == 2

    sample = dataset[0]
    assert (sample.img == np.zeros([4, 4, 3])).all()
    assert sample.height == sample.width == 4
