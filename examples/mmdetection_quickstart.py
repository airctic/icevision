from icevision.all import *

data_dir = icedata.fridge.load_data()
class_map = icedata.fridge.class_map()

parser = icedata.fridge.parser(data_dir, class_map=class_map)

train_records, valid_records = parser.parse()

presize, size = 256, 128

train_tfms = tfms.A.Adapter(
    [*tfms.A.aug_tfms(size=size, presize=presize), tfms.A.Normalize()]
)
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

train_dl = mmdetection_models.train_dl(train_ds, batch_size=2, shuffle=True)
valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=2, shuffle=False)

### MODEL

from mmcv import Config

cfg = Config.fromfile(
    "~/git/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
)

from mmdet.models import build_detector

model = build_detector(cfg.model, cfg.train_cfg, cfg.test_cfg)


class LitModel(mmdetection_models.lightning.ModelAdapter):
    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-3)


lit_model = LitModel(model)
trainer = pl.Trainer(
    max_epochs=10, gpus=1, num_sanity_val_steps=0, check_val_every_n_epoch=99
)
trainer.fit(lit_model, train_dl, valid_dl)
