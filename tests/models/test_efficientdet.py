from mantisshrimp.imports import *
from mantisshrimp import *

img = 255 * np.ones((4, 4, 3), dtype=np.uint8)
bboxes = [BBox.from_xyxy(1, 2, 3, 4), BBox.from_xyxy(4, 3, 2, 1)]
labels = [1, 2]


## Build a single sample
def build_train_sample(img: np.ndarray, labels: List[int], bboxes: List[BBox]):
    x = im2tensor(img)

    y = {
        "cls": tensor(labels, dtype=torch.int64),
        "boxes": tensor([bbox.yxyx for bbox in bboxes], dtype=torch.float64),
    }

    return x, y


image, target = build_train_sample(img=img, labels=labels, bboxes=bboxes)

assert torch.all(image == torch.ones((3, 4, 4), dtype=torch.float32))

assert torch.all(target["cls"] == tensor([1, 2], dtype=torch.int64))
assert torch.all(
    target["boxes"] == tensor([[2, 1, 4, 3], [3, 4, 1, 2]], dtype=torch.float64)
)

## Dataloader


def collate_fn(samples):
    train_samples = [build_train_sample(**sample) for sample in samples]
    return tuple(zip(*train_samples))


def dataloader(dataset, **kwargs):
    return DataLoader(dataset, collate_fn=collate_fn, **kwargs)


dataset = [{"img": img, "labels": labels, "bboxes": bboxes}] * 2

dl = dataloader(dataset, batch_size=2)
xb, yb = first(dl)

assert len(xb) == len(yb) == 2

for x in xb:
    assert x.shape == (3, 4, 4)

for y in yb:
    assert set(y.keys()) == {"cls", "boxes"}
