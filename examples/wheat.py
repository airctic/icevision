from mantisshrimp.all import *
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

catmap = CategoryMap([Category(0, "wheat")])
parser = DataParser(df, source, catmap=catmap, info_parser=WheatInfoParser, annotation_parser=WheatAnnotationParser)

train_rs, valid_rs = parser.parse()

tfm = AlbuTransform([A.Flip(p=0.8), A.ShiftScaleRotate(p=0.8, scale_limit=(0, 0.5))])

train_ds = Dataset(train_rs, tfm)
valid_ds = Dataset(valid_rs)

train_dl = RCNNDataLoader(train_ds, batch_size=4, num_workers=8)
valid_dl = RCNNDataLoader(valid_ds, batch_size=4, num_workers=8)

items = [train_ds[0] for _ in range(2)]
grid2([partial(show_item, o, label=False) for o in items], show=True)

metrics = [COCOMetric(valid_rs, catmap)]

model = MantisFasterRCNN(2)
model.prepare_optimizers(SGD, 1e-3)

from mantisshrimp.models.utils import *

adfdf
model.model_splits
len(list(model.trainable_params_splits()))

# Get trainable parameters from parameter groups
# If a parameter group does not have trainable params, it does not get added
params = []
for pg in model.model_splits:
    ps = list(filter_params(pg, only_trainable=True))
    if ps: params.append(ps)



a = list(filter_params(model.model_splits[0], only_trainable=True))
a

unfreeze(filter_params(model))
freeze(model.parameters())

for group in model.model_splits[:2]:
    freeze(filter_params(group))

model.get_lrs(slice(1e-5, 1e-3))

# list((model.param_groups[-1][0]).parameters())[0].requires_grad
list((model.model_splits[-1].box_head.fc6).parameters())[0].requires_grad
model.model_splits[-1]

params = list(filter_params(model, only_trainable=False))
len(params)
params

unfreeze(params for params in model.model_splits[:None])
len(model.model_splits[:None])
model.model_splits[0]
len(model.model_splits)

trainer = Trainer(max_epochs=1, gpus=1)
trainer.fit(model, train_dl, valid_dl)

rs = random.choices(valid_rs, k=2)
ims, preds = model.predict(rs=rs)
show_preds(ims, preds)


class WheatModel(MantisFasterRCNN):
    def configure_optimizers(self):
        opt = SGD(params(self), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        sched = OneCycleLR(opt, max_lr=1e-3, total_steps=len(train_dl), pct_start=0.3)
        return [opt], [{"scheduler": sched, "interval": "step"}]

