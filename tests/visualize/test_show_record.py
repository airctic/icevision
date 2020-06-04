from mantisshrimp.imports import plt
from mantisshrimp import *


def test_show_record(record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    show_record(record)
    plt.show()
