from mantisshrimp.all import *
import scipy.io

data_dir = Path("/home/lgvaz/data/birds")

# images_dir = data_dir / "images"
# annotations_dir = data_dir / "annotations-mat"

# img_files = get_image_files(images_dir)
# mat_files = get_files(annotations_dir, extensions=[".mat"])

# img_file = img_files[0]
# img = open_img(img_file)
# show_img(img, show=True)

# mat_file = annotations_dir.ls()[0].ls()[0]
# mat = scipy.io.loadmat(str(mat_file))
# len(mat_files)

# mat.keys()
# mask = MaskArray(mat["seg"])

# mat["bbox"]
# mat["bbox"][0, 0]

# show_annotation(img, masks=mask[None], show=True)
# plt.show()

# annotations_dir.ls()
# data_dir.ls()

# img_file.stem
# mat_file.stem

# mat["bbox"][0, 0].astype(int)


class ImageParser(parsers.Parser, parsers.FilepathMixin):
    def __init__(self, data_dir):
        self.image_filepaths = get_image_files(data_dir)

    def __iter__(self) -> Any:
        yield from self.image_filepaths

    def filepath(self, o) -> Union[str, Path]:
        return o

    def imageid(self, o) -> Hashable:
        return o.stem


image_parser = ImageParser(data_dir)

# records = image_parser.parse()[0]


class BirdMaskFile(MaskFile):
    def to_mask(self, h, w):
        mat = scipy.io.loadmat(str(self.filepath))
        return MaskArray(mat["seg"])[None]


class AnnotationParser(
    parsers.Parser, parsers.MasksMixin, parsers.BBoxesMixin, parsers.LabelsMixin
):
    def __init__(self, data_dir, class_map):
        self.mat_filepaths = get_files(
            data_dir / "annotations-mat", extensions=[".mat"]
        )
        self.class_map = class_map

    def __iter__(self) -> Any:
        yield from self.mat_filepaths

    def masks(self, o) -> List[Mask]:
        return [BirdMaskFile(o)]

    def bboxes(self, o) -> List[BBox]:
        mat = scipy.io.loadmat(str(o))
        bbox = mat["bbox"]
        xyxy = [int(bbox[pos]) for pos in ["left", "top", "right", "bottom"]]
        return [BBox.from_xyxy(*xyxy)]

    def imageid(self, o) -> Hashable:
        return o.stem

    def labels(self, o) -> List[int]:
        class_name = o.parent.name
        return [self.class_map.get_name(class_name)]


classes_file = data_dir / "lists/classes.txt"
classes = classes_file.read().splitlines()
class_map = ClassMap(classes)

annotation_parser = AnnotationParser(data_dir, class_map)
parser = parsers.CombinedParser(image_parser, annotation_parser)

data_splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(data_splitter)

presize, size = 512, 384
train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size, presize), tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])

train_ds = Dataset(train_records[:100], train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

# samples = [train_ds[456] for _ in range(6)]
# show_samples(samples, ncols=3, denormalize_fn=denormalize_imagenet, show=True)

train_dl = mask_rcnn.train_dataloader(
    train_ds, batch_size=16, num_workers=4, shuffle=True
)
valid_dl = mask_rcnn.valid_dataloader(valid_ds, batch_size=16, num_workers=4)

model = mask_rcnn.model(num_classes=len(class_map))

metrics = [COCOMetric(COCOMetricType.mask)]
learn = mask_rcnn.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)

# learn.freeze()
# learn.lr_find()

learn.fine_tune(2, 3e-3)

