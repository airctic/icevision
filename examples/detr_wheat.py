import pandas as pd
from mantisshrimp.imports import *
from mantisshrimp.hub.detr import *


class WheatParser(DetrBBoxParser):
    def __init__(self, df, source):
        self.df = df
        self.source = source
        self.imageid_map = IDMap()

    def __iter__(self):
        yield from self.df.itertuples()

    def __len__(self):
        return len(self.df)

    def prepare(self, o):
        self.bbox = BBox.from_xywh(*np.fromstring(o.bbox[1:-1], sep=","))

    def imageid(self, o) -> int:
        return self.imageid_map[o.image_id]

    def filepath(self, o) -> Union[str, Path]:
        return self.source / f"{o.image_id}.jpg"

    def height(self, o) -> int:
        return o.height

    def width(self, o) -> int:
        return o.width

    def label(self, o) -> List[int]:
        return [1]

    def bbox(self, o) -> List[BBox]:
        return [self.bbox]

    def area(self, o) -> List[float]:
        return [self.bbox.area]

    def iscrowd(self, o) -> List[bool]:
        return [0]


def get_datasets(args):
    # parse records
    source = Path(args.data_path)
    df = pd.read_csv(source / "train.csv")
    data_splitter = RandomSplitter([0.8, 0.2])
    parser = WheatParser(df, source / "train")
    train_rs, valid_rs = parser.parse(data_splitter)
    # We use the transforms defined by the authors
    train_tfm = detr_transform("train")
    valid_tfm = detr_transform("val")
    train_dataset = DetrDataset(train_rs, train_tfm)
    valid_dataset = DetrDataset(valid_rs, valid_tfm)
    return train_dataset, valid_dataset


if __name__ == "__main__":
    # adds new arguments to original args_parser
    args_parser = get_args_parser()
    args_parser.add_argument("--data_path", type=str)
    args = args_parser.parse_args()

    train_dataset, valid_dataset = get_datasets(args)
    run_detr(args=args, dataset_train=train_dataset, dataset_val=valid_dataset)
