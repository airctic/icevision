from icevision.all import *

# import torch.distributed as dist

# dist.init_process_group(
#     "gloo", init_method="file:///tmp/somefile", rank=0, world_size=1
# )

data_url = "https://s3.amazonaws.com/fast-ai-sample/camvid_tiny.tgz"
data_dir = icedata.load_data(data_url, "camvid_tiny") / "camvid_tiny"
codes = np.loadtxt(data_dir / "codes.txt", dtype=str)
class_map = ClassMap(list(codes))
images_dir = data_dir / "images"
labels_dir = data_dir / "labels"
image_files = get_image_files(images_dir)
records = RecordCollection(SemanticSegmentationRecord)

for image_file in pbar(image_files):
    record = records.get_by_record_id(image_file.stem)

    if record.is_new:
        record.set_filepath(image_file)
        record.set_img_size(get_img_size(image_file))
        record.segmentation.set_class_map(class_map)

    mask_file = SemanticMaskFile(labels_dir / f"{image_file.stem}_P.png")
    record.segmentation.set_mask(mask_file)

records = records.autofix()
train_records, valid_records = records.make_splits(RandomSplitter([0.8, 0.2]))

# TODO: Get this from the config?
presize, size = 520, 480
presize, size = ImgSize(presize, int(presize * 0.75)), ImgSize(size, int(size * 0.75))

aug_tfms = tfms.A.aug_tfms(
    presize=presize,
    size=size,
    pad=None,
    crop_fn=partial(tfms.A.RandomCrop, p=0.5),
    shift_scale_rotate=tfms.A.ShiftScaleRotate(rotate_limit=2),
)
train_tfms = tfms.A.Adapter([*aug_tfms, tfms.A.Normalize()])
valid_tfms = tfms.A.Adapter([tfms.A.resize(size), tfms.A.Normalize()])

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

model_type = models.mmseg.deeplabv3
backbone = model_type.backbones.resnet50_d8

train_dl = model_type.train_dl(train_ds, batch_size=8, num_workers=0, shuffle=True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=8, num_workers=0, shuffle=False)

model = model_type.model(
    backbone=backbone(pretrained=False), num_classes=class_map.num_classes
)


def accuracy_camvid(pred, target):
    # ignores void pixels
    keep_idxs = target != class_map.get_by_name("Void")
    target = target[keep_idxs]
    pred = pred.argmax(dim=1)[keep_idxs]

    return (pred == target).float().mean()


learn = model_type.fastai.learner(
    dls=[train_dl, valid_dl],
    model=model,
    metrics=[accuracy_camvid],
    splitter=fastai.trainable_params,
)

# learn.lr_find()
learn.fine_tune(3, 1e-4)