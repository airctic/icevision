from icevision.utils import get_image_size


def test_get_image_size(samples_source):
    filepath = samples_source / "voc/JPEGImages/2007_000063.jpg"
    size = get_image_size(filepath)

    assert size == (500, 375)
