from icevision.parsers.coco_parser import COCOKeypointsMetadata
from icevision.all import *


def test_draw_record(coco_record, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    img = draw_record(coco_record, display_bbox=False)
    show_img(img, show=True)


def test_draw_sample(fridge_ds, fridge_class_map, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    sample = fridge_ds[0][0]
    img = draw_sample(
        sample, class_map=fridge_class_map, denormalize_fn=denormalize_imagenet
    )
    show_img(img, show=True)


def test_draw_pred():
    img = np.zeros((200, 200, 3))
    pred = {"bboxes": [BBox.from_xywh(100, 100, 50, 50)], "labels": [1]}
    pred_img = draw_pred(img=img, pred=pred)

    assert (pred_img[101, 101] != [0, 0, 0]).all()


def test_draw_keypoints(keypoints_img_128372):
    img = np.zeros((427, 640, 3))
    color = (np.random.random(3) * 0.6 + 0.4) * 255
    kps = KeyPoints.from_xyv(keypoints_img_128372, COCOKeypointsMetadata)
    img = draw_keypoints(img=img, kps=kps, color=color)
    assert (img[0][0] == np.array([0, 0, 0])).all()
