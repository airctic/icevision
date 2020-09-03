from mantisshrimp.all import *


def test_draw_record(record):
    img = draw_record(record, display_bbox=False)
    show_img(img, show=True)


def test_draw_sample(fridge_ds):
    sample = fridge_ds[0][0]
    class_map = datasets.fridge.class_map()
    img = draw_sample(sample, class_map=class_map, denormalize_fn=denormalize_imagenet)
    show_img(img, show=True)


def test_draw_pred(record):
    img = np.zeros((200, 200, 3))
    pred = {"bboxes": [BBox.from_xywh(100, 100, 50, 50)], "labels": [1]}
    pred_img = draw_pred(img=img, pred=pred)

    assert (pred_img[101, 101] != [0, 0, 0]).all()
