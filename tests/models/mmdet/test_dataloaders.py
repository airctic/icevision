import pytest
from icevision.all import *


@pytest.fixture()
def img():
    return 255 * np.ones((4, 4, 3), dtype=np.uint8)


@pytest.fixture()
def labels():
    return [1, 2]


@pytest.fixture()
def bboxes():
    return [BBox.from_xyxy(1, 2, 3, 4), BBox.from_xyxy(10, 20, 30, 40)]


@pytest.fixture()
def masks():
    return MaskArray(np.ones((2, 4, 4), dtype=np.uint8))


@pytest.fixture()
def class_map():
    return ClassMap(classes=["1", "2"])


@pytest.fixture()
def mask_records(img, labels, bboxes, masks, class_map):
    Record = create_mixed_record(
        (
            ImageRecordMixin,
            LabelsRecordMixin,
            BBoxesRecordMixin,
            MasksRecordMixin,
            ClassMapRecordMixin,
        )
    )
    record = Record()
    record.set_imageid(1)
    record.set_img(img)
    record.add_labels(labels)
    record.add_bboxes(bboxes)
    record.add_masks([masks])
    record.set_class_map(class_map)
    sample = record.load()
    return [sample] * 2


def test_bbox_dataloader(fridge_ds, fridge_class_map):
    train_ds, _ = fridge_ds
    train_dl = models.mmdet.faster_rcnn.train_dl(
        train_ds, batch_size=2, num_workers=0, shuffle=False
    )
    data, recs = first(train_dl)

    assert set(data.keys()) == {"gt_bboxes", "gt_labels", "img", "img_metas"}
    assert ((np.array(recs[0]["labels"]) - 1) == data["gt_labels"][0].numpy()).all()
    assert ((np.array(recs[1]["labels"]) - 1) == data["gt_labels"][1].numpy()).all()
    assert data["img_metas"][0]["img_shape"] == (384, 384, 3)
    assert data["img_metas"][1]["pad_shape"] == (384, 384, 3)
    assert (
        data["img_metas"][1]["scale_factor"] == np.array([1.0, 1.0, 1.0, 1.0])
    ).all()
    assert recs[0]["class_map"] == fridge_class_map
    assert recs[1]["class_map"] == fridge_class_map
    assert recs[1]["img"].shape == data["img"][1].numpy().swapaxes(0, -1).shape


def _test_common_mmdet_mask_batch(batch):
    data, recs = batch

    assert set(data.keys()) == {
        "gt_bboxes",
        "gt_labels",
        "gt_masks",
        "img",
        "img_metas",
    }
    assert ((np.array(recs[0]["labels"]) - 1) == data["gt_labels"][0].numpy()).all()
    assert data["img_metas"][0]["img_shape"] == (4, 4, 3)
    assert (
        data["img_metas"][0]["scale_factor"] == np.array([1.0, 1.0, 1.0, 1.0])
    ).all()
    assert "BitmapMasks" in str(type(data["gt_masks"][0]))
    assert len(data["gt_masks"]) == 2
    assert data["gt_masks"][0].height == data["gt_masks"][0].width == 4
    assert set(recs[0].keys()) == {
        "bboxes",
        "class_map",
        "height",
        "imageid",
        "img",
        "labels",
        "masks",
        "width",
    }


def test_mmdet_mask_rcnn_build_train_batch(mask_records):
    batch = models.mmdet.mask_rcnn.build_train_batch(mask_records)
    _test_common_mmdet_mask_batch(batch)


def test_mmdet_mask_rcnn_build_valid_batch(mask_records):
    batch = models.mmdet.mask_rcnn.build_valid_batch(mask_records)
    _test_common_mmdet_mask_batch(batch)


def test_mmdet_mask_rcnn_build_infer_batch(mask_records):
    batch = models.mmdet.mask_rcnn.build_infer_batch(mask_records)
    data, recs = batch

    assert set(data.keys()) == {
        "img",
        "img_metas",
    }
    assert recs[0]["labels"] == [1, 2]
    assert data["img_metas"][0][0]["img_shape"] == (4, 4, 3)
    assert (
        data["img_metas"][0][0]["scale_factor"] == np.array([1.0, 1.0, 1.0, 1.0])
    ).all()
    assert set(recs[0].keys()) == {
        "bboxes",
        "class_map",
        "height",
        "imageid",
        "img",
        "labels",
        "masks",
        "width",
    }
