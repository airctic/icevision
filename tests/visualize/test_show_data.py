from mantisshrimp.all import *


def test_show_record(record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    show_record(record, display_bbox=False, show=True)


def test_show_sample(sample, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    show_sample(sample, show=True)


def test_show_pred(record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    img = np.zeros((200, 200, 3))
    pred = {"bboxes": [BBox.from_xywh(100, 100, 50, 50)], "labels": [1]}
    show_pred(img=img, pred=pred)
