# import pytest
# from icevision.parsers.coco_parser import COCOKeypointsMetadata
# from icevision.all import *

# TODO bugfix/1135
# def test_draw_record(coco_record, monkeypatch):
#     monkeypatch.setattr(plt, "show", lambda: None)
#     img = draw_record(coco_record, display_bbox=False)
#     show_img(img, show=True)


# @pytest.mark.parametrize(
#     [
#         "color_map",
#         "include_only",
#         "exclude_labels",
#         "include_instances_task_names",
#         "include_classification_task_names",
#     ],
#     [
#         (None, None, [], False, False),
#         (None, ["fridge"], [], False, False),
#         (None, None, ["fridge"], False, False),
#         (None, None, [], True, False),
#         (None, None, [], False, True),
#         (
#             {
#                 "background": "#fff",
#                 "can": "#faa",
#                 "carton": "black",
#                 "milk_bottle": [255, 255, 255],
#                 "water_bottle": (255, 255, 255),
#             },
#             None,
#             [],
#             False,
#             False,
#         ),
#     ],
# )
# def test_draw_sample(
#     fridge_ds,
#     fridge_class_map,
#     monkeypatch,
#     color_map,
#     include_only,
#     exclude_labels,
#     include_instances_task_names,
#     include_classification_task_names,
# ):
#     monkeypatch.setattr(plt, "show", lambda: None)
#     sample = fridge_ds[0][0]
#     img = draw_sample(
#         sample,
#         class_map=fridge_class_map,
#         denormalize_fn=denormalize_imagenet,
#         color_map=color_map,
#         include_only=include_only,
#         exclude_labels=exclude_labels,
#     )
#     show_img(img, show=True)


# @pytest.mark.skip
# def test_draw_pred():
#     img = np.zeros((200, 200, 3))
#     pred = {"bboxes": [BBox.from_xywh(100, 100, 50, 50)], "labels": [1]}
#     pred_img = draw_pred(img=img, pred=pred)

#     assert (pred_img[101, 101] != [0, 0, 0]).all()


# def test_draw_keypoints(keypoints_img_128372):
#     img = np.zeros((427, 640, 3)).astype(np.uint8)
#     color = (np.random.random(3) * 0.6 + 0.4) * 255
#     kps = KeyPoints.from_xyv(keypoints_img_128372, COCOKeypointsMetadata)
#     img = draw_keypoints(img=img, kps=kps, color=color)
#     assert (img[0][0] == np.array([0, 0, 0])).all()
