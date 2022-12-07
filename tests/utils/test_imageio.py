import pytest
from icevision.all import *
from icevision.utils.imageio import open_gray_scale_image


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("images/test_3_bands_int8.tif", (100, 101, 3)),
        ("images/test_3_bands_int16.tif", (100, 101, 3)),
        ("images/test_3_bands_int32.tif", (100, 101, 3)),
        ("images/test_3_bands_float32.tif", (100, 101, 3)),
    ],
)
def test_open_img_returns_tiff_with_shape_h_w_c_when_tiff_has_3_bands_with_gray_is_false(
    samples_source, fn, expected
):

    image = open_img(samples_source / fn, gray=False)

    assert image.shape == expected


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("images/test_3_bands_int8.tif", (100, 101)),
        ("images/test_3_bands_int16.tif", (100, 101)),
        ("images/test_3_bands_int32.tif", (100, 101)),
        ("images/test_3_bands_float32.tif", (100, 101)),
    ],
)
def test_open_img_returns_tiff_with_shape_h_w_c_when_tiff_has_3_bands_with_gray_is_true(
    samples_source, fn, expected
):

    image = open_img(samples_source / fn, gray=True)

    assert image.shape == expected


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("voc/JPEGImages/2007_000063.jpg", (375, 500, 3)),
        ("images2/flies.jpeg", (3888, 2592, 3)),
    ],
)
def test_open_img_returns_image_such_that_its_numpy_shape_would_be_h_w_c_when_reading_rgb_image_with_gray_is_false(
    samples_source, fn, expected
):

    image = open_img(samples_source / fn, gray=False)

    assert np.array(image).shape == expected


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("voc/JPEGImages/2007_000063.jpg", (375, 500)),
        ("images2/flies.jpeg", (3888, 2592)),
    ],
)
def test_open_img_returns_image_such_that_its_numpy_shape_would_be_h_w_when_reading_rgb_image_with_gray_is_true(
    samples_source, fn, expected
):

    image = open_img(samples_source / fn, gray=True)

    assert np.array(image).shape == expected


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("gray_scale/gray_scale_h_10_w_10_image.tiff", (10, 10)),
        ("gray_scale/gray_scale_h_50_w_50_image.tiff", (50, 50)),
        ("gray_scale/test_1_bands_int8.tif", (100, 101)),
        ("gray_scale/test_1_bands_int16.tif", (100, 101)),
        ("gray_scale/test_1_bands_int32.tif", (100, 101)),
        ("gray_scale/test_1_bands_float32.tif", (100, 101)),
    ],
)
def test_open_img_returns_image_with_shape_h_w_when_reading_grayscale_tiff_with_gray_is_false(
    samples_source, fn, expected
):

    image = open_img(samples_source / fn, gray=False)

    assert image.shape == expected


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("gray_scale/gray_scale_h_10_w_10_image.tiff", (10, 10)),
        ("gray_scale/gray_scale_h_50_w_50_image.tiff", (50, 50)),
        ("gray_scale/test_1_bands_int8.tif", (100, 101)),
        ("gray_scale/test_1_bands_int16.tif", (100, 101)),
        ("gray_scale/test_1_bands_int32.tif", (100, 101)),
        ("gray_scale/test_1_bands_float32.tif", (100, 101)),
    ],
)
def test_open_img_returns_image_with_shape_h_w_when_reading_grayscale_tiff_with_gray_is_true(
    samples_source, fn, expected
):

    image = open_img(samples_source / fn, gray=True)

    assert image.shape == expected


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("gray_scale/grayscale_int8.png", (200, 150, 3)),
    ],
)
def test_open_img_returns_image_such_that_its_numpy_shape_would_be_h_w_c_when_reading_grayscale_image_with_gray_is_false(
    samples_source, fn, expected
):

    image = open_img(samples_source / fn, gray=False)

    assert np.array(image).shape == expected


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("gray_scale/grayscale_int8.png", (200, 150)),
    ],
)
def test_open_img_returns_image_such_that_its_numpy_shape_would_be_h_w_when_reading_grayscale_image_with_gray_is_true(
    samples_source, fn, expected
):

    image = open_img(samples_source / fn, gray=True)

    assert np.array(image).shape == expected


@pytest.mark.parametrize(
    "fn",
    [
        ("voc/JPEGImages/2007_000063.jpg"),
        ("images2/flies.jpeg"),
        ("gray_scale/grayscale_int8.png"),
    ],
)
def test_open_img_returns_PIL_image_when_image_not_tiff(samples_source, fn):

    image = open_img(samples_source / fn)

    assert isinstance(image, PIL.Image.Image)


@pytest.mark.parametrize(
    "fn",
    [
        ("gray_scale/test_1_bands_int8.tif"),
        ("gray_scale/test_1_bands_int16.tif"),
        ("gray_scale/test_1_bands_int32.tif"),
        ("gray_scale/test_1_bands_float32.tif"),
        ("images/test_3_bands_int8.tif"),
        ("images/test_3_bands_int16.tif"),
        ("images/test_3_bands_int32.tif"),
        ("images/test_3_bands_float32.tif"),
    ],
)
def test_open_img_returns_numpy_array_when_image_is_tiff(samples_source, fn):

    image = open_img(samples_source / fn)

    assert isinstance(image, np.ndarray)


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("voc/JPEGImages/2007_000063.jpg", (375, 500)),
        ("images2/flies.jpeg", (3888, 2592)),
        ("gray_scale/grayscale_int8.png", (200, 150)),
        ("gray_scale/test_1_bands_int8.tif", (100, 101)),
        ("gray_scale/test_1_bands_int16.tif", (100, 101)),
        ("gray_scale/test_1_bands_int32.tif", (100, 101)),
        ("gray_scale/test_1_bands_float32.tif", (100, 101)),
        ("images/test_3_bands_int8.tif", (100, 101)),
        ("images/test_3_bands_int16.tif", (100, 101)),
        ("images/test_3_bands_int32.tif", (100, 101)),
        ("images/test_3_bands_float32.tif", (100, 101)),
    ],
)
def test_get_img_size_size_returns_size_in_format_w_h(samples_source, fn, expected):

    size = get_img_size(samples_source / fn)

    assert size.width == expected[1]
    assert size.height == expected[0]


@pytest.mark.parametrize(
    "fn",
    [
        ("gray_scale/grayscale_int8.png"),
        ("gray_scale/gray_scale_h_10_w_10_image.tiff"),
        ("gray_scale/gray_scale_h_50_w_50_image.tiff"),
        ("gray_scale/test_1_bands_int8.tif"),
        ("gray_scale/test_1_bands_int16.tif"),
        ("gray_scale/test_1_bands_int32.tif"),
        ("gray_scale/test_1_bands_float32.tif"),
    ],
)
def test_open_gray_scale_image_returns_numpy_array(samples_source, fn):

    image = open_gray_scale_image(samples_source / fn)

    assert isinstance(image, np.ndarray)


@pytest.mark.parametrize(
    "fn",
    [
        ("gray_scale/grayscale_int8.png"),
        ("gray_scale/gray_scale_h_10_w_10_image.tiff"),
        ("gray_scale/gray_scale_h_50_w_50_image.tiff"),
        ("gray_scale/test_1_bands_int8.tif"),
        ("gray_scale/test_1_bands_int16.tif"),
        ("gray_scale/test_1_bands_int32.tif"),
        ("gray_scale/test_1_bands_float32.tif"),
    ],
)
def test_open_gray_scale_image_returns_array_with_3_stacked_grayscale_channels(
    samples_source, fn
):

    image = open_gray_scale_image(samples_source / fn)

    assert image.shape[-1] == 3


@pytest.mark.parametrize(
    "fn",
    [
        ("gray_scale/grayscale_int8.png"),
        ("gray_scale/gray_scale_h_10_w_10_image.tiff"),
        ("gray_scale/gray_scale_h_50_w_50_image.tiff"),
        ("gray_scale/test_1_bands_int8.tif"),
        ("gray_scale/test_1_bands_int16.tif"),
        ("gray_scale/test_1_bands_int32.tif"),
        ("gray_scale/test_1_bands_float32.tif"),
    ],
)
def test_open_gray_scale_image_returns_stacked_channels_with_same_values(
    samples_source, fn
):

    image = open_gray_scale_image(samples_source / fn)

    assert (image[:, :, 0] == image[:, :, 1]).all()
    assert (image[:, :, 0] == image[:, :, 2]).all()
    assert (image[:, :, 1] == image[:, :, 2]).all()


def test_get_img_size_from_data_returns_img_size_when_data_is_pil_img():
    expected_height = random.randrange(10, 1000)
    expected_width = random.randrange(10, 1000)
    data = np.random.randint(0, 256, (expected_height, expected_width), dtype=np.uint8)
    data = PIL.Image.fromarray(data)

    img_size = get_img_size_from_data(data)

    img_size == ImgSize(width=expected_width, height=expected_height)


def test_get_img_size_from_data_returns_img_size_when_data_is_numpy_array_with_2_dimensions():
    expected_height = random.randrange(10, 1000)
    expected_width = random.randrange(10, 1000)
    data = np.random.randint(0, 256, (expected_height, expected_width), dtype=np.uint8)

    img_size = get_img_size_from_data(data)

    img_size == ImgSize(width=expected_width, height=expected_height)


def test_get_img_size_from_data_returns_img_size_when_data_is_numpy_array_with_3_dimensions():
    expected_height = random.randrange(10, 1000)
    expected_width = random.randrange(10, 1000)
    channels = 3
    data = np.random.randint(
        0, 256, (expected_height, expected_width, channels), dtype=np.uint8
    )

    img_size = get_img_size_from_data(data)

    img_size == ImgSize(width=expected_width, height=expected_height)


def test_get_img_size_from_data_raises_an_exception_when_data_is_not_pil_img_or_numpy_array():
    data = "some data"

    with pytest.raises(Exception) as _:
        get_img_size_from_data(data)
