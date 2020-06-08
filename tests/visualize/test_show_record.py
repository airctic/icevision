import pytest
from mantisshrimp.imports import plt
from mantisshrimp import *


def test_show_record_label_bbox_mask(record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    show_record(record)
    plt.show()


def test_show_record_label_bbox(record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    record = record.copy()
    record.pop("mask")
    show_record(record)
    plt.show()


def test_show_record_label_mask(record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    record = record.copy()
    record.pop("bbox")
    show_record(record)
    plt.show()


def test_show_record_label(record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    record = record.copy()
    record.pop("bbox")
    record.pop("mask")
    with pytest.raises(ValueError) as e:
        show_record(record)
    assert str(e.value) == "Can only display labels if bboxes or masks are given"
    plt.show()
