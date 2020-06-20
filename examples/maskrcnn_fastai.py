from fastai2.vision.all import *
from fastai2.metrics import Metric as FastaiMetric
from fastai2.data.load import DataLoader as FastaiDataLoader
from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.hub.pennfundan import *


# TODO: auto convert datalaoder (just need to grab collate_fn)
# TODO: auto convert metric


class FastaiMetricAdapter(FastaiMetric):
    def __init__(self, metric: Metric):
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def accumulate(self, learn: Learner):
        self.metric.accumulate(*learn.xb, *learn.yb, learn.pred)

    @property
    def value(self):
        return self.metric.finalize()


def zip_convert(t):
    raise NotImplementedError


# Unified setup
source = get_pennfundan_data()
parser = PennFundanParser(source)
splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)
train_dataset = Dataset(train_records)
valid_dataset = Dataset(valid_records)
model = MantisMaskRCNN(num_classes=2)
train_dataloader = model.dataloader(train_dataset, batch_size=2, num_workers=2)
valid_dataloader = model.dataloader(valid_dataset, batch_size=2, num_workers=2)
###


def convert_dataloader_to_fastai(dataloader: DataLoader):
    def raise_error_convert(data):
        raise NotImplementedError

    class FastaiDataLoaderWithCollate(FastaiDataLoader):
        def create_batch(self, b):
            return (dataloader.collate_fn, raise_error_convert)[self.prebatched](b)

    # extract shuffle from the type of sampler used
    if isinstance(dataloader.sampler, SequentialSampler):
        shuffle = False
    elif isinstance(dataloader.sampler, RandomSampler):
        shuffle = True
    else:
        raise ValueError(
            f"Sampler {type(dataloader.sampler)} not supported. Fastai only"
            "supports RandomSampler or SequentialSampler"
        )

    return FastaiDataLoaderWithCollate(
        dataset=dataloader.dataset,
        bs=dataloader.batch_size,
        num_workers=dataloader.num_workers,
        drop_last=dataloader.drop_last,
        shuffle=shuffle,
        pin_memory=dataloader.pin_memory,
    )


class RCNNCallback(Callback):
    def begin_batch(self):
        assert len(self.xb) == len(self.yb) == 1, "Only works for single input-output"
        self.learn.xb = (self.xb[0], self.yb[0])
        self.learn.yb = ()

    def after_pred(self):
        self.learn.yb = [self.learn.xb[1]]
        self.learn.xb = [self.learn.xb[0]]

    def begin_validate(self):
        # put model in training mode so we can calculate losses for validation
        self.model.train()

    def after_loss(self):
        if not self.training:
            self.model.eval()
            self.learn.pred = self.model(*self.xb)
            self.model.train()


def rcnn_learner(dls: DataLoaders, model: MantisRCNN, cbs=None, **kwargs):
    cbs = [RCNNCallback()] + L(cbs)

    def model_splitter(model):
        return model.params_splits()

    learn = Learner(
        dls=dataloaders,
        model=model,
        loss_func=model.loss,
        cbs=cbs,
        metrics=metrics,
        splitter=model_splitter,
    )

    # HACK: patch AvgLoss (in original, find_bs gives errors)
    class RCNNAvgLoss(AvgLoss):
        def accumulate(self, learn):
            bs = len(learn.yb)
            self.total += to_detach(learn.loss.mean()) * bs
            self.count += bs

    recorder = [cb for cb in learn.cbs if isinstance(cb, Recorder)][0]
    recorder.loss = RCNNAvgLoss()

    return learn


metric = COCOMetric(valid_records, bbox=True, mask=True)
metric = FastaiMetricAdapter(metric)

train_dataloader2 = convert_dataloader_to_fastai(train_dataloader)
valid_dataloader2 = convert_dataloader_to_fastai(valid_dataloader)
# TODO: Check if cuda is available, see how fastai does it
dataloaders = DataLoaders(train_dataloader2, valid_dataloader2).to(torch.device("cuda"))

learn = rcnn_learner(dls=dataloaders, model=model)

learn.fit_one_cycle(3, lr=2e-4)

# TODO: add some tests
# check that model_splits is freezing the correct layers
