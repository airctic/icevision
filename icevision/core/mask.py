__all__ = ["Mask", "MaskArray", "MaskFile", "VocMaskFile", "RLE", "Polygon"]

from icevision.imports import *
from icevision.utils import *
from PIL import Image


class Mask(ABC):
    @abstractmethod
    def to_mask(self, h, w) -> "MaskArray":
        pass

    @abstractmethod
    def to_erle(self, h, w):
        pass


# TODO: Assert shape? (bs, height, width)
@dataclass
class MaskArray(Mask):
    data: np.ndarray

    def __post_init__(self):
        self.data = self.data.astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return type(self)(self.data[i])

    def to_tensor(self):
        return tensor(self.data, dtype=torch.uint8)

    def to_mask(self, h, w):
        return self

    def to_erle(self, h, w):
        return mask_utils.encode(np.asfortranarray(self.data.transpose(1, 2, 0)))

    def to_coco_rle(self, h, w) -> List[dict]:
        """From https://stackoverflow.com/a/49547872/6772672"""
        assert self.data.shape[1:] == (h, w)
        rles = []
        for mask in self.data:
            counts = []
            flat = itertools.groupby(mask.ravel(order="F"))
            for i, (value, elements) in enumerate(flat):
                if i == 0 and value == 1:
                    counts.append(0)
                counts.append(len(list(elements)))
            rles.append({"counts": counts, "size": (h, w)})
        return rles

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def from_masks(cls, masks, h, w):
        new = []
        # TODO: Instead of if checks, RLE and Polygon can return with extra dim
        for o in masks:
            m = o.to_mask(h, w).data
            if isinstance(o, (RLE, Polygon)):
                new.append(m[None])
            elif isinstance(o, MaskFile):
                new.append(m)
            elif isinstance(o, MaskArray):
                new.append(m)
            else:
                raise ValueError(f"Segmented type {type(o)} not supported")
        return cls(np.concatenate(new))


@dataclass
class MaskFile(Mask):
    filepath: Union[str, Path]

    def __post_init__(self):
        self.filepath = Path(self.filepath)

    def to_mask(self, h, w):
        mask = open_img(self.filepath, gray=True)
        obj_ids = np.unique(mask)[1:]
        masks = mask == obj_ids[:, None, None]
        return MaskArray(masks)

    def to_coco_rle(self, h, w) -> List[dict]:
        return self.to_mask(h=h, w=w).to_coco_rle(h=h, w=w)

    def to_erle(self, h, w):
        return self.to_mask(h, w).to_erle(h, w)


@dataclass
class VocMaskFile(MaskFile):
    """Extension of `MaskFile` for VOC masks.
    Removes the color pallete and optionally drops void pixels.

    Args:
          drop_void (bool): drops the void pixels, which should have the value 255.
    """

    drop_void: bool = True

    def to_mask(self, h, w) -> MaskArray:
        mask_arr = np.array(Image.open(self.filepath))
        obj_ids = np.unique(mask_arr)[1:]
        masks = mask_arr == obj_ids[:, None, None]

        if self.drop_void:
            masks = masks[:-1, ...]

        return MaskArray(masks)


@dataclass(frozen=True)
class RLE(Mask):
    counts: List[int]

    def to_mask(self, h, w):
        "From https://www.kaggle.com/julienbeaulieu/imaterialist-detectron2"
        mask = np.full(h * w, 0, dtype=np.uint8)
        for start, ones in zip(self.counts[::2], self.counts[1::2]):
            # counting starts on one
            start -= 1
            if ones:
                mask[start : start + ones] = 1
        mask = mask.reshape((h, w), order="F")
        return MaskArray(mask)

    def to_coco(self) -> List[int]:
        coco_counts, total = [], 0
        for start, ones in zip(self.counts[::2], self.counts[1::2]):
            zeros = start - total - 1
            coco_counts.extend([zeros, ones])
            total = start + ones - 1
        # don't include last count if it's zero
        if coco_counts[-1] == 0:
            coco_counts = coco_counts[:-1]
        return coco_counts

    def to_erle(self, h, w):
        return mask_utils.frPyObjects([{"counts": self.counts, "size": [h, w]}], h, w)

    @classmethod
    def from_string(cls, s, sep=" "):
        return cls(lmap(int, s.split(sep)))

    @classmethod
    def from_kaggle(cls, counts):
        """Described [here](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/overview/evaluation)"""
        if len(counts) % 2 != 0:
            raise ValueError("Counts must be divisible by 2")
        return cls(counts)

    @classmethod
    def from_coco(cls, counts):
        """Described [here](https://stackoverflow.com/a/49547872/6772672)"""
        # when counts is odd, round it with 0 ones at the end
        if len(counts) % 2 != 0:
            counts = counts + [0]

        kaggle_counts, total = [], 0
        for zeros, ones in zip(counts[::2], counts[1::2]):
            start = zeros + total + 1
            kaggle_counts.extend([start, ones])
            total += zeros + ones
        return cls.from_kaggle(kaggle_counts)


@dataclass(frozen=True)
class Polygon(Mask):
    points: List[List[int]]

    def to_mask(self, h, w):
        erle = self.to_erle(h=h, w=w)
        mask = mask_utils.decode(erle).sum(axis=-1)  # Sum is for unconnected polygons
        assert mask.max() == 1, "Probable overlap in polygons"
        return MaskArray(mask)

    def to_erle(self, h, w):
        return mask_utils.frPyObjects(self.points, h, w)
