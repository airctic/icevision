from icevision.all import *

selection = 1

if selection == 0:
    model_type = models.mmdet.mask_rcnn
    backbone = model_type.backbones.resnet50_fpn_1x

if selection == 1:
    model_type = models.mmdet.mask_rcnn
    backbone = model_type.backbones.mask_rcnn_swin_t_p4_w7_fpn_1x_coco

if selection == 2:
    model_type = models.mmdet.yolact
    backbone = model_type.backbones.r101_1x8_coco

# Loading Data
data_dir = icedata.pennfudan.load_data()
parser = icedata.pennfudan.parser(data_dir)

# train_ds, valid_ds = icedata.pennfudan.dataset(data_dir)
train_rs, valid_rs = parser.parse()

# Transforms
image_size = 512
train_tfms = tfms.A.Adapter(
    [*tfms.A.aug_tfms(size=image_size, presize=1024), tfms.A.Normalize()]
)
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(image_size), tfms.A.Normalize()])

train_ds = Dataset(train_rs, train_tfms)
valid_ds = Dataset(valid_rs, valid_tfms)

# DataLoaders
train_dl = model_type.train_dl(train_ds, batch_size=4, num_workers=0, shuffle=True)
valid_dl = model_type.valid_dl(valid_ds, batch_size=4, num_workers=0, shuffle=False)

# valid_batch = first(valid_dl)

# show batch
# model_type.show_batch(first(train_dl), ncols=4)

infer_ds = Dataset(valid_rs, valid_tfms)

infer_dl = model_type.infer_dl(infer_ds[0:10], batch_size=4, shuffle=False)

model = model_type.model(
    backbone=backbone(pretrained=True), num_classes=parser.class_map.num_classes
)

preds = model_type.predict_from_dl(model, infer_dl, keep_images=True)

show_preds(preds[0:4])

a = 1
