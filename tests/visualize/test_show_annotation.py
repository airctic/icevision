from icevision.all import *


def test_show_annotation(record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    data = record.load()
    show_annotation(
        img=data["img"],
        labels=data["labels"],
        bboxes=data["bboxes"],
        masks=data["masks"],
    )
    plt.show()
