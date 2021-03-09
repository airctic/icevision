import pytest
from icevision.all import *


def test_show_record(coco_record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    show_record(coco_record, display_bbox=False, show=True)


def test_show_sample(coco_sample, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    show_sample(coco_sample, show=True)


def test_show_keypoints_sample(coco_keypoints_parser, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    records = coco_keypoints_parser.parse(data_splitter=SingleSplitSplitter())[0]
    sample = records[1].load()
    show_sample(sample, denormalize_fn=None)


def test_show_keypoints_record(coco_keypoints_parser, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    records = coco_keypoints_parser.parse(data_splitter=SingleSplitSplitter())[0]
    show_record(records[0])


@pytest.mark.skip
def test_show_pred(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    img = np.zeros((200, 200, 3))
    pred = {"bboxes": [BBox.from_xywh(100, 100, 50, 50)], "labels": [1]}
    show_pred(img=img, pred=pred)
