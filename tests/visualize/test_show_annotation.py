from mantisshrimp.imports import plt
from mantisshrimp import *


def test_show_annotation(record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    data = default_prepare_record(record)
    show_annotation(
        data["img"], labels=data["labels"], bboxes=data["bboxes"], masks=data["masks"]
    )
    plt.show()
