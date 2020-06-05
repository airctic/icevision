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


class WheatModel(MantisFasterRCNN):
    def __init__(
        self,
        n_class: int,
        parser: Parser,
        batch_size: int,
        train_tfm: Transform = None,
        valid_tfm: Transform = None,
        num_workers=0,
        **kwargs,
    ):
        super().__init__(n_class, **kwargs)
        self.parser = parser
        self.batch_size = batch_size
        self.train_tfm = train_tfm
        self.valid_tfm = valid_tfm
        self.num_workers = num_workers

    def configure_optimizers(self):
        opt = SGD(self.parameters(), 1e-3, momentum=0.9)
        return opt

    def prepare_data(self) -> None:
        data_splitter = RandomSplitter([0.8, 0.2])
        train_rs, valid_rs = self.parser.parse(data_splitter)
        self.train_ds = Dataset(train_rs, self.train_tfm)
        self.valid_ds = Dataset(valid_rs, self.valid_tfm)

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader(
            self.valid_ds, batch_size=self.batch_size, num_workers=self.num_workers
        )


source = Path("/home/lgvaz/.data/wheat")
df = pd.read_csv(source / "train.csv")
parser = WheatParser(df, source / "train")
train_tfm = AlbuTransform([A.Flip()])

model = WheatModel(2, parser, train_tfm=train_tfm, batch_size=4, num_workers=8)


trainer = Trainer(max_epochs=1, gpus=1)
trainer.fit(model)
