from mantisshrimp.all import *


def test_show_record_label_bbox_mask(record, monkeypatch):
    # monkeypatch.setattr(plt, "show", lambda: None)
    show_record(record, display_bbox=False)
    plt.show()


def show_record(
    record,
    class_map: Optional[ClassMap] = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_mask: bool = True,
    ax: plt.Axes = None,
    show: bool = False,
    prepare_record: Optional[callable] = None,
) -> None:
    img = draw_record(
        record=record,
        class_map=class_map,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
        prepare_record=prepare_record,
    )
    show_img(img=img, ax=ax, show=show)

