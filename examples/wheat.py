from mantisshrimp.imports import *
from mantisshrimp import *
import pandas as pd, albumentations as A

source = Path("/home/lgvaz/.data/wheat")
df = pd.read_csv(source / "train.csv")


class WheatInfoParser(InfoParser):
    def filepath(self, o):
        return self.source / f"train/{o.image_id}.jpg"

    def imageid(self, o):
        return o.image_id

    def h(self, o):
        return o.height

    def w(self, o):
        return o.width

    def __iter__(self):
        yield from self.data.itertuples()


class WheatAnnotationParser(AnnotationParser):
    def imageid(self, o):
        return o.image_id

    def label(self, o):
        return 0

    def bbox(self, o):
        return BBox.from_xywh(*np.fromstring(o.bbox[1:-1], sep=","))

    def __iter__(self):
        yield from df.itertuples()


class WheatModel(MantisFasterRCNN):
    def __init__(
        self, n_freeze_epochs, n_unfreeze_epochs, freeze_lrs, unfreeze_lrs, **kwargs
    ):
        self.n_freeze_epochs = n_freeze_epochs
        self.n_unfreeze_epochs = n_unfreeze_epochs
        self.freeze_lrs = freeze_lrs
        self.unfreeze_lrs = unfreeze_lrs
        super().__init__(**kwargs)

    def configure_optimizers(self):
        param_groups = self.get_optimizer_param_groups()
        opt = SGD(param_groups, momentum=0.9)
        sched = OneCycleLR(opt, 1e-3, 10)
        return [opt], [sched]

    def on_epoch_start(self):
        if self.current_epoch == 0:
            self.freeze_to(-1)
            self.replace_sched(self.n_freeze_epochs, self.freeze_lrs)

        if self.current_epoch == self.n_freeze_epochs:
            self.freeze_to(0)
            self.replace_sched(self.n_unfreeze_epochs, self.unfreeze_lrs)

    def replace_sched(self, n_epochs, lrs):
        total_steps = len(self.train_dataloader()) * n_epochs
        opt = self.trainer.optimizers[0]
        sched = OneCycleLR(opt, lrs, total_steps)
        sched = {"scheduler": sched, "interval": "step"}
        scheds = self.trainer.configure_schedulers([sched])
        # Replace scheduler
        self.trainer.lr_schedulers = scheds
        lr_logger.on_train_start(self.trainer, self)


catmap = CategoryMap([Category(0, "wheat")])
parser = DataParser(
    df,
    source,
    catmap=catmap,
    info_parser=WheatInfoParser,
    annotation_parser=WheatAnnotationParser,
)

train_rs, valid_rs = parser.parse()

tfm = AlbuTransform([A.Flip(p=0.8), A.ShiftScaleRotate(p=0.8, scale_limit=(0, 0.5))])

train_ds = Dataset(train_rs, tfm)
valid_ds = Dataset(valid_rs)

# train_dl = RCNNDataLoader(train_ds, batch_size=4, num_workers=8)
# valid_dl = RCNNDataLoader(valid_ds, batch_size=4, num_workers=8)

items = [train_ds[0] for _ in range(2)]
grid2([partial(show_item, o, label=False) for o in items], show=True)

# Cannot do, we don't know the model before hand
# n_param_groups = len(list(model.params_splits()))
n_param_groups = 8
freeze_lrs = [1e-3] * n_param_groups
unfreeze_lrs = np.linspace(5e-6, 5e-4, n_param_groups).tolist()

metrics = [COCOMetric(valid_rs, catmap)]
model = WheatModel(
    n_freeze_epochs=1,
    n_unfreeze_epochs=3,
    freeze_lrs=freeze_lrs,
    unfreeze_lrs=unfreeze_lrs,
    n_class=2,
    # metrics=metrics,
)

train_dl = model.dataloader(dataset=train_ds, batch_size=4, num_workers=8)
valid_dl = model.dataloader(dataset=valid_ds, batch_size=4, num_workers=8)

lr_logger = LearningRateLogger()
trainer = Trainer(max_epochs=4, gpus=1, weights_summary=None, callbacks=[lr_logger])
trainer.fit(model, train_dl, valid_dl)
