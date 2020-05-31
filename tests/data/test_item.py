def test_item(item):
    assert item.imageid == 0
    assert item.img.shape == (427, 640, 3)
    assert item.masks.shape == (16, 427, 640)
    assert item.labels == [6, 1, 1, 1, 1, 1, 1, 1, 27, 27, 1, 3, 27, 1, 27, 27]
    assert item.iscrowds == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert item.bboxes[0].xyxy == [0, 73.89, 416.44, 379.02]
