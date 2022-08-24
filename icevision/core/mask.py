__all__ = [
    "Mask",
    "MaskArray",
    "MaskFile",
    "VocMaskFile",
    "RLE",
    "Polygon",
    "EncodedRLEs",
    "SemanticMaskFile",
]

from icevision.imports import *
from icevision.utils import *
from PIL import Image

from icevision.utils.imageio import open_gray_scale_image


class Mask(ABC):
    @abstractmethod
    def to_mask(self, h, w) -> "MaskArray":
        pass

    @abstractmethod
    def to_erles(self, h, w) -> "EncodedRLEs":
        pass


class EncodedRLEs(Mask):
    def __init__(self, erles: List[dict] = None):
        print("EncodedRLEs::__init__")
        print(f"erles: {erles}")
        print(f"len erles: {len(erles)}")
        self.erles = erles or []

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self)} objects>"

    def __len__(self):
        return len(self.erles)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.erles == other.erles
        return False

    def append(self, v: "EncodedRLEs"):
        print("EncodedRLEs::append")
        self.erles.extend(v.erles)

    def extend(self, v: List["EncodedRLEs"]):
        print("EncodedRLEs::extend")

        for o in v:
            self.append(o)

    def pop(self, i: int):
        print("EncodedRLEs::pop")

        self.erles.pop(i)

    def to_mask(self, h, w) -> "MaskArray":
        print("EncodedRLEs::to_mask")
        print(f"erles len : {len(self.erles)}")

        mask = mask_utils.decode(self.erles)

        print(f"mask_utils.decode(self.erles) : {mask}")
        print(f"mask_utils.decode(self.erles).shape : {mask.shape}")  # (h,w,c)

        mask = mask.transpose(1, 0, 2)  # (h,w,c) => (w,h,c)

        print(f"Mask new shape {mask.shape}")
        print("Creating MaskArray(mask)")
        return MaskArray(mask)

    def to_erles(self, h, w) -> "EncodedRLEs":
        print("EncodedRLEs::to_erles")

        return self


class MaskArray(Mask):
    """Binary numpy array representation of a mask.

    # Arguments
        data: Mask array, with the dimensions: (width, height, num_instances)
        pad_dim: bool
    """

    def __init__(self, data: np.uint8, pad_dim: bool = True):
        print("MaskArray::__init__")
        print(f"data.shape RAW {data.shape}")

        if pad_dim and (len(data.shape) == 2):
            data = np.expand_dims(data, 0)
        self.data = data.astype(np.uint8)
        print(f"data.shape {data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return type(self)(self.data[i])

    def to_tensor(self):
        print(f"MaskArray::to_tensor")
        print(f"self.data.shape {self.data.shape}")

        return tensor(self.data, dtype=torch.uint8)

    def to_mask(self, h, w):
        print("MaskArray::to_mask")

        return self

    def to_erles(self, h, w) -> EncodedRLEs:
        print("MaskArray::to_erles")
        print(f"self.data.shape {self.data.shape}")

        # HACK: force empty annotations to have valid shape
        if len(self.data.shape) == 1:

            data_with_new_axes = self.data[:, None, None]

            print(f"data_with_new_axes.shape {data_with_new_axes.shape}")

            data_with_new_axes = data_with_new_axes.transpose(1, 2, 0)

            print(f"transposed data_with_new_axes.shape {data_with_new_axes.shape}")

            fortranarray_mask = np.asfortranarray(data_with_new_axes)

            print(f"fortranarray_mask.shape {fortranarray_mask.shape}")

            encoded_mask = mask_utils.encode(fortranarray_mask)

            print("encoded_mask {encoded_mask}")

            return EncodedRLEs(encoded_mask)
        else:

            transposed_data = self.data.transpose(1, 2, 0)

            print(f"transposed data.shape {transposed_data.shape}")

            fortranarray_mask = np.asfortranarray(transposed_data)

            print(f"fortranarray_mask.shape {fortranarray_mask.shape}")

            encoded_mask = mask_utils.encode(fortranarray_mask)

            print(f"encoded_mask {encoded_mask}")

            return EncodedRLEs(encoded_mask)

    def to_coco_rle(self, h, w) -> List[dict]:
        """From https://stackoverflow.com/a/49547872/6772672"""
        print(f"MaskArray::to_coco_rle (h={h}, w={w})")
        print(f"self.data.shape {self.data.shape}")

        c_h_w_data = self.data.transpose(2, 1, 0)

        assert c_h_w_data.shape[1:] == (h, w)

        rles = []
        for mask in c_h_w_data:

            print(f"mask.shape {mask.shape}")

            counts = []
            flat = itertools.groupby(mask.ravel(order="F"))

            print(f"flat {flat}")

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
    def from_masks(cls, masks: Union[EncodedRLEs, Sequence[Mask]], h: int, w: int):
        print(f"MaskArray::from_masks class method (h={h}, w={w})")
        # HACK: check for backwards compatibility
        if isinstance(masks, EncodedRLEs):
            print(f"masks is EncodedRLEs")
            print(f"masks.to_mask(h=h, w=w)")

            return masks.to_mask(h=h, w=w)
        else:
            print(f"masks is NOT EncodedRLEs")

            masks_arrays = [o.to_mask(h=h, w=w).data for o in masks]
            if len(masks_arrays) == 0:
                return cls(np.array([]))
            else:
                return cls(np.concatenate(masks_arrays))


class MaskFile(Mask):
    """Holds the path to mask image file.

    # Arguments
        filepath: Path to the mask image file.
    """

    def __init__(self, filepath: Union[str, Path]):
        print("MaskFile::__init__")
        self.filepath = Path(filepath)

    def to_mask(self, h=None, w=None):

        print(f"MaskFile::to_mask(h={h}, w={w})")
        print(f"Opeining '{self.filepath}'")

        mask_img = open_img(self.filepath, gray=True)

        print(f"mask_img.shape '{mask_img.shape}'")

        if (h is not None) and (w is not None):
            # If the dimensions provided in h and w do not match the size of the mask, resize the mask accordingly
            org_img_size = get_img_size_from_data(mask_img)

            print(f"org_img_size '{org_img_size}'")

            # TODO: Check NEAREST is always the best option or only for binary?
            if org_img_size.width != w or org_img_size.height != h:
                print(f"Resizing mask ...")

                mask_img = mask_img.resize(
                    (w, h), resample=PIL.Image.NEAREST
                )  # TODO bugfix/1135 not always PIL anymore

        mask = image_to_numpy(mask_img)
        print(f"mask.shape {mask.shape}")
        obj_ids = np.unique(mask)[1:]
        masks = mask == obj_ids[:, None, None]

        return MaskArray(masks)

    def to_coco_rle(self, h, w) -> List[dict]:
        print(f"MaskFile::to_coco_rle")

        return self.to_mask(h=h, w=w).to_coco_rle(h=h, w=w)

    def to_erles(self, h, w) -> EncodedRLEs:
        print(f"MaskFile::to_erles")

        return self.to_mask(h=h, w=w).to_erles(h=h, w=w)


class VocMaskFile(MaskFile):
    """Extension of `MaskFile` for VOC masks.
    Removes the color pallete and optionally drops void pixels.

    # Arguments
        drop_void (bool): drops the void pixels, which should have the value 255.
        filepath: Path to the mask image file.
    """

    def __init__(self, filepath: Union[str, Path], drop_void: bool = True):
        print(f"VocMaskFile::__init__")

        super().__init__(filepath=filepath)
        self.drop_void = drop_void

    def to_mask(self, h, w) -> MaskArray:

        print(f"VocMaskFile::to_mask(h={h}, w={w})")
        print(f"Opening '{self.filepath}'...")

        img = open_img(self.filepath, gray=True, ensure_no_data_convert=True)

        mask_arr = image_to_numpy(img)

        print(f"mask_arr.shape {mask_arr.shape}")
        print(f"OLD mask_arr.shape {np.array(Image.open(self.filepath)).shape}")

        obj_ids = np.unique(mask_arr)[1:]
        print(f"obj_ids[:, None, None] {obj_ids[:, None, None]}")
        masks = mask_arr == obj_ids[:, None, None]

        if self.drop_void:
            masks = masks[..., :-1]

        print(f"mask_arr.shape after drop void {masks.shape}")

        return MaskArray(masks)


class RLE(Mask):
    """Run length encoding of a mask.

    Don't instantiate this class directly, instead use the classmethods
    `from_coco` and `from_kaggle`.
    """

    def __init__(self, counts: List[int]):
        print(f"RLE::__init__")

        self.counts = counts

    def to_mask(self, h, w) -> "MaskArray":
        print(f"RLE::to_mask(h={h}, w={w})")

        return self.to_erles(h=h, w=w).to_mask(h=h, w=w)

    def to_coco(self) -> List[int]:
        print(f"RLE::to_coco")

        return self.counts

    def to_erles(self, h, w) -> EncodedRLEs:

        print(f"RLE::to_erles(h={h}, w={w})")

        return EncodedRLEs(
            mask_utils.frPyObjects([{"counts": self.to_coco(), "size": [h, w]}], h, w)
        )

    @classmethod
    def from_string(cls, s, sep=" "):
        return cls(lmap(int, s.split(sep)))

    @classmethod
    def from_kaggle(cls, counts: Sequence[int]):
        """Described [here](https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/overview/evaluation)"""
        if len(counts) % 2 != 0:
            raise ValueError("Counts must be divisible by 2")

        current = 1
        coco_counts = []
        for start, count in zip(counts[::2], counts[1::2]):
            coco_counts.append(start - current)  # zeros
            coco_counts.append(count)  # ones
            current = start + count

        # remove trailing zero
        if coco_counts[-1] == 0:
            coco_counts.pop(-1)

        return cls.from_coco(coco_counts)

    @classmethod
    def from_coco(cls, counts: Sequence[int]):
        """Described [here](https://stackoverflow.com/a/49547872/6772672)"""
        return cls(counts)


class Polygon(Mask):
    """Polygon representation of a mask

    # Arguments
        points: The vertices of the polygon in the COCO standard format.
    """

    def __init__(self, points: List[List[int]]):
        print(f"Polygon::__init__")

        self.points = points

    def to_mask(self, h, w):
        print(f"Polygon::to_mask")

        return self.to_erles(h=h, w=w).to_mask(h=h, w=w)

    def to_erles(self, h, w) -> EncodedRLEs:
        print(f"Polygon::to_erles")

        erles = mask_utils.frPyObjects(self.points, h, w)
        erle = mask_utils.merge(erles)  # make unconnected polygons a single mask
        return EncodedRLEs([erle])


class SemanticMaskFile(Mask):
    """Holds the path to mask image file.

    # Arguments
        filepath: Path to the mask image file.
    """

    def __init__(self, filepath: Union[str, Path], binary=False):
        print(f"SemanticMaskFile::__init__")

        self.filepath = Path(filepath)
        self.binary = binary

    def to_mask(self, h, w, pad_dim=True):

        print(f"SemanticMaskFile::to_mask")
        print(f"Opening '{self.filepath}'...")

        # TODO: convert the 255 masks
        mask = open_img(self.filepath, gray=True)

        print(f"mask.shape {mask}")

        # If the dimensions provided in h and w do not match the size of the mask, resize the mask accordingly
        org_img_size = get_img_size_from_data(mask)
        print(f"org_img_size {org_img_size}")

        # TODO: Check NEAREST is always the best option or only for binary?
        if org_img_size.width != w or org_img_size.height != h:
            print(f"Resizing...")
            mask = mask.resize((w, h), resample=PIL.Image.NEAREST)

        mask = image_to_numpy(mask)

        print(f"mask.shape {mask.shape}")

        # convert 255 pixels to 1
        if self.binary:
            mask[mask == 255] = 1

        # control array padding behaviour
        return MaskArray(mask, pad_dim=pad_dim)

    def to_coco_rle(self, h, w) -> List[dict]:
        raise NotImplementedError

    def to_erles(self, h, w) -> EncodedRLEs:
        # HACK: Doesn't make sense to convert to ERLE?
        return self
