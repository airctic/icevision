import pytest
from icevision.all import *


@pytest.fixture
def dummy_class_map():
    return ClassMap(["dummy-1", "dummy-2"], background=None)


@pytest.fixture
def dummy_class_map_elaborate():
    return ClassMap(["dummy-1", "dummy-2", "dummy-3", "dummy-4"], background=None)


def test_classification_multilabel(dummy_class_map):
    rec = BaseRecord([ClassificationLabelsRecordComponent(is_multilabel=True)])
    rec.classification.set_class_map(dummy_class_map)
    rec.classification.set_labels_by_id([0, 1])

    assert rec.classification.label_ids == [0, 1]
    assert (rec.classification.one_hot_encoded() == np.array([1, 1])).all()


@pytest.mark.parametrize(
    "label_ids",
    [
        ([0, 1]),
        ([0]),
    ],
)
def test_classification_single_label(dummy_class_map, label_ids):
    rec = BaseRecord([ClassificationLabelsRecordComponent(is_multilabel=False)])
    rec.classification.set_class_map(dummy_class_map)
    rec.classification.set_labels_by_id(label_ids)

    if len(label_ids) > 1:
        # label_ids == [0, 1]
        # Setting two labels when `is_multilabel=False` raises an error
        with pytest.raises(AutofixAbort):
            rec.classification._autofix()
    else:
        # label_ids == [0]
        # Only one label must be assigned
        assert all(rec.classification._autofix().values())
        assert rec.classification.one_hot_encoded().sum() == 1


@pytest.mark.parametrize(
    "label_ids",
    [
        ([0, 1, 2]),
        ([0, 1]),
        ([0]),
    ],
)
def test_one_hot_encodings(dummy_class_map_elaborate, label_ids):
    rec = BaseRecord([ClassificationLabelsRecordComponent(is_multilabel=True)])
    rec.classification.set_class_map(dummy_class_map_elaborate)
    rec.classification.set_labels_by_id(label_ids)

    assert all(rec.classification._autofix().values())

    # Ensure we have the correct no. of labels and that they are indeed
    # one-hot encoded
    one_hot_values = rec.classification.one_hot_encoded()
    assert one_hot_values.sum() == len(label_ids)
    assert np.unique(one_hot_values).tolist() == [0, 1]


class FakeComposite:
    def __init__(self) -> None:
        self.is_set_img_size_called = False
        self.is_img_set_as_original = False
        self.img_size_received = None

    def set_img_size(self, img_size, original):
        self.is_set_img_size_called = True
        self.is_img_set_as_original = original
        self.img_size_received = img_size


@pytest.mark.parametrize(
    "fn",
    [
        ("voc/JPEGImages/2007_000063.jpg"),
        ("images2/flies.jpeg"),
        ("images/test_3_bands_int8.tif"),
        ("images/test_3_bands_int16.tif"),
        ("images/test_3_bands_int32.tif"),
        ("images/test_3_bands_float32.tif"),
    ],
)
def test_filepath_record_component_can_load_rgb_image_when_gray_is_false(
    samples_source, fn
):
    rec = FilepathRecordComponent(gray=False)
    rec.set_filepath(samples_source / fn)
    rec.composite = FakeComposite()

    rec._load()


@pytest.mark.parametrize(
    "fn",
    [
        ("voc/JPEGImages/2007_000063.jpg"),
        ("images2/flies.jpeg"),
        ("images/test_3_bands_int8.tif"),
        ("images/test_3_bands_int16.tif"),
        ("images/test_3_bands_int32.tif"),
        ("images/test_3_bands_float32.tif"),
    ],
)
def test_filepath_record_component_can_load_rgb_image_when_gray_is_true(
    samples_source, fn
):
    rec = FilepathRecordComponent(gray=True)
    rec.set_filepath(samples_source / fn)
    rec.composite = FakeComposite()

    rec._load()


@pytest.mark.parametrize(
    "fn",
    [
        ("voc/JPEGImages/2007_000063.jpg"),
        ("images2/flies.jpeg"),
        ("images/test_3_bands_int8.tif"),
        ("images/test_3_bands_int16.tif"),
        ("images/test_3_bands_int32.tif"),
        ("images/test_3_bands_float32.tif"),
    ],
)
def test_filepath_record_component_sets_img_size_during_load(samples_source, fn):
    rec = FilepathRecordComponent(gray=False)
    rec.set_filepath(samples_source / fn)
    rec.composite = FakeComposite()

    rec._load()

    assert rec.composite.is_set_img_size_called == True


@pytest.mark.parametrize(
    "fn",
    [
        ("gray_scale/grayscale_int8.png"),
        ("gray_scale/test_1_bands_int8.tif"),
        ("gray_scale/test_1_bands_int16.tif"),
        ("gray_scale/test_1_bands_int32.tif"),
        ("gray_scale/test_1_bands_float32.tif"),
    ],
)
def test_filepath_record_component_can_load_grayscale_image_when_gray_is_true(
    samples_source, fn
):
    rec = FilepathRecordComponent(gray=True)
    rec.set_filepath(samples_source / fn)
    rec.composite = FakeComposite()

    rec._load()


@pytest.mark.parametrize(
    "fn",
    [
        ("gray_scale/grayscale_int8.png"),
        ("gray_scale/test_1_bands_int8.tif"),
        ("gray_scale/test_1_bands_int16.tif"),
        ("gray_scale/test_1_bands_int32.tif"),
        ("gray_scale/test_1_bands_float32.tif"),
    ],
)
def test_grayscale_record_component_can_load_grayscale_image(samples_source, fn):
    rec = GrayScaleRecordComponent()
    rec.composite = FakeComposite()
    rec.set_filepath(samples_source / fn)

    rec._load()


def test_image_record_component_set_img_sets_img_received_as_input_when_its_a_numpy_array():
    rec = ImageRecordComponent()
    rec.composite = FakeComposite()
    input_img = np.random.randint(0, 256, (10, 10), dtype=np.uint8)

    rec.set_img(img=input_img)

    assert (rec.img == input_img).all()


def test_image_record_component_set_img_sets_img_received_as_input_when_its_a_pil_img():
    rec = ImageRecordComponent()
    rec.composite = FakeComposite()
    input_img = PIL.Image.fromarray(np.random.rand(10, 10))

    rec.set_img(img=input_img)

    assert rec.img == input_img


def test_image_record_component_set_img_sets_img_size():
    rec = ImageRecordComponent()
    rec.composite = FakeComposite()
    expected_width = random.randint(10, 100)
    expected_height = random.randint(10, 100)
    input_img = np.random.randint(
        0, 256, (expected_width, expected_height), dtype=np.uint8
    )

    rec.set_img(img=input_img)

    assert rec.composite.img_size_received == ImgSize(
        width=expected_width, height=expected_height
    )


def test_image_record_component_set_img_as_original_image():
    rec = ImageRecordComponent()
    rec.composite = FakeComposite()
    expected_width = random.randint(10, 100)
    expected_height = random.randint(10, 100)
    input_img = np.random.randint(
        0, 256, (expected_width, expected_height), dtype=np.uint8
    )

    rec.set_img(img=input_img)

    assert rec.composite.is_img_set_as_original == True
