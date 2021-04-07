from icevision.all import *

data_dir = icedata.pennfudan.load_data()

parser = icedata.pennfudan.parser(data_dir)

train_records, valid_records = parser.parse(autofix=False, background_id=-1)

show_record(train_records[0], show=True)

presize, size = 256, 128

train_tfms = tfms.A.Adapter(
    [*tfms.A.aug_tfms(size=size, presize=presize), tfms.A.Normalize()]
)
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])

train_ds = Dataset(train_records, train_tfms)
valid_ds = Dataset(valid_records, valid_tfms)

train_dl = mmdet.train_dl(train_ds, batch_size=2, shuffle=True)
valid_dl = mmdet.valid_dl(valid_ds, batch_size=2, shuffle=False)
