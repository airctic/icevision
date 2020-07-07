import pytest
from mantisshrimp.imports import *
from mantisshrimp import *


@pytest.fixture()
def img():
    return 255 * np.ones((4, 4, 3), dtype=np.uint8)


@pytest.fixture()
def labels():
    return [1, 2]


@pytest.fixture()
def bboxes():
    return [BBox.from_xyxy(1, 2, 3, 4), BBox.from_xyxy(4, 3, 2, 1)]


class EfficientDet:
    @staticmethod
    def build_train_sample(img: np.ndarray, labels: List[int], bboxes: List[BBox]):
        x = im2tensor(img)

        y = {
            "cls": tensor(labels, dtype=torch.int64),
            "boxes": tensor([bbox.yxyx for bbox in bboxes], dtype=torch.float64),
        }

        return x, y

    @classmethod
    def collate_fn(cls, samples):
        train_samples = [cls.build_train_sample(**sample) for sample in samples]
        return tuple(zip(*train_samples))

    @classmethod
    def dataloader(cls, dataset, **kwargs):
        return DataLoader(dataset=dataset, collate_fn=cls.collate_fn, **kwargs)


def test_efficient_det_build_train_sample(img, labels, bboxes):
    image, target = EfficientDet.build_train_sample(
        img=img, labels=labels, bboxes=bboxes
    )

    assert torch.all(image == torch.ones((3, 4, 4), dtype=torch.float32))

    assert torch.all(target["cls"] == tensor([1, 2], dtype=torch.int64))
    assert torch.all(
        target["boxes"] == tensor([[2, 1, 4, 3], [3, 4, 1, 2]], dtype=torch.float64)
    )


def test_efficient_det_dataloader(img, labels, bboxes):
    dataset = [{"img": img, "labels": labels, "bboxes": bboxes}] * 2

    dl = EfficientDet.dataloader(dataset, batch_size=2)
    xb, yb = first(dl)

    assert len(xb) == len(yb) == 2

    for x in xb:
        assert x.shape == (3, 4, 4)

    for y in yb:
        assert set(y.keys()) == {"cls", "boxes"}
