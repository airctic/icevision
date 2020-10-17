"""
Copied from https://github.com/scardine/image_size/blob/master/get_image_size.py
:Name:        get_image_size
:Purpose:     extract image dimensions given a file path
:Author:      Paulo Scardine (based on code from Emmanuel VAÏSSE)
:Created:     26/09/2013
:Copyright:   (c) Paulo Scardine 2013
:Licence:     MIT
"""
import io
import pytest
from icevision.utils.get_image_size import (
    UnknownImageFormat,
    get_image_size,
    get_image_metadata,
    get_image_metadata_from_bytesio,
    get_image_size_from_bytesio,
)

image_fields = ["path", "type", "file_size", "width", "height"]


@pytest.fixture
def data(samples_source):
    return [
        {
            "path": str(samples_source / "voc/JPEGImages/2007_000063.jpg"),
            "width": 500,
            "height": 375,
            "file_size": 126171,
            "type": "JPEG",
        }
    ]


def test_get_image_size_from_bytesio(data):
    img = data[0]

    p = img["path"]
    with io.open(p, "rb") as fp:
        b = fp.read()
    fp = io.BytesIO(b)
    sz = len(b)
    output = get_image_size_from_bytesio(fp, sz)

    assert output == (img["width"], img["height"])


def test_get_image_metadata_from_bytesio(data):
    img = data[0]

    p = img["path"]
    with io.open(p, "rb") as fp:
        b = fp.read()
    fp = io.BytesIO(b)
    sz = len(b)
    output = get_image_metadata_from_bytesio(fp, sz)

    for field in image_fields:
        assert getattr(output, field) == None if field == "path" else img[field]


def test_get_image_metadata(data):
    img = data[0]
    output = get_image_metadata(img["path"])

    for field in image_fields:
        assert getattr(output, field) == img[field]


def test_get_image_metadata__ENOENT_OSError():
    with pytest.raises(OSError):
        get_image_metadata("THIS_DOES_NOT_EXIST")


def test_get_image_metadata__not_an_image_UnknownImageFormat():
    with pytest.raises(UnknownImageFormat):
        get_image_metadata(__file__)


def test_get_image_size(data):
    img = data[0]

    output = get_image_size(img["path"])

    assert output == (img["width"], img["height"])
