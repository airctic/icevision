from mantisshrimp.imports import *
from mantisshrimp import *
import pandas as pd
import albumentations as A


class WheatParser(ImageInfoParser, FasterRCNNParser):
    def __init__(self, df, source):
        self.df = df
        self.source = source
        self.imageid_map = IDMap()

    def __iter__(self):
        yield from self.df.itertuples()

    def __len__(self):
        return len(self.df)

    def imageid(self, o) -> int:
        return self.imageid_map[o.image_id]

    def filepath(self, o) -> Union[str, Path]:
        return self.source / f"{o.image_id}.jpg"

    def height(self, o) -> int:
        return o.height

    def width(self, o) -> int:
        return o.width

    def label(self, o):
        return 1

    def bbox(self, o) -> BBox:
        return BBox.from_xywh(*np.fromstring(o.bbox[1:-1], sep=","))


source = Path("/home/lgvaz/.data/wheat")
df = pd.read_csv(source / "train.csv")

parser = WheatParser(df, source / "train")
data_splitter = RandomSplitter([0.8, 0.2])
train_rs, valid_rs = parser.parse(data_splitter)

tfm = AlbuTransform([A.Flip()])
train_ds = Dataset(train_rs, tfm)
valid_ds = Dataset(valid_rs)


class WheatModel(MantisFasterRCNN):
    def configure_optimizers(self):
        opt = SGD(self.parameters(), 1e-3, momentum=0.9)
        return opt


model = WheatModel(2)

train_dl = model.dataloader(dataset=train_ds, batch_size=4, shuffle=True, num_workers=8)
valid_dl = model.dataloader(dataset=valid_ds, batch_size=4, num_workers=8)

trainer = Trainer(max_epochs=1, gpus=1)

trainer.fit(model, train_dl, valid_dl)
