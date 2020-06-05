__all__ = ["MaskArray", "MaskFile", "RLE", "Polygon"]

from ..imports import *
from ..utils import *


class Mask(ABC):
    @abstractmethod
    def to_mask(self, h, w) -> "MaskArray":
        pass

    @abstractmethod
    def to_erle(self, h, w):
        pass


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

    def to_erle(self, h, w):
        return self.to_mask(h, w).to_erle(h, w)


@dataclass(frozen=True)
class RLE(Mask):
    counts: List[int]

    def to_mask(self, h, w):
        "From https://www.kaggle.com/julienbeaulieu/imaterialist-detectron2"
        mask = np.full(h * w, 0, dtype=np.uint8)
        for i, start_pixel in enumerate(self.counts[::2]):
            mask[start_pixel : start_pixel + self.counts[2 * i + 1]] = 1
        mask = mask.reshape((h, w), order="F")
        return MaskArray(mask)

    def to_erle(self, h, w):
        raise NotImplementedError("Convert counts to coco style")
        # return mask_utils.frPyObjects([{'counts':self.counts, 'size':[h,w]}], h, w)

    @classmethod
    def from_string(cls, s, sep=" "):
        return cls(lmap(int, s.split(sep)))

    @classmethod
    def from_kaggle(cls, counts):
        "Described [here](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/overview/evaluation)"
        if len(counts) % 2 != 0:
            raise ValueError("Counts must be divisible by 2")
        return cls(counts)

    @classmethod
    def from_coco(cls, counts):
        "Described [here](https://stackoverflow.com/a/49547872/6772672)"
        kaggle_counts, total = [], 0
        for zrs, ons in zip(counts[::2], counts[1::2]):
            kaggle_counts.extend([zrs + total + 1, ons])
            total += zrs + ons
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
