from mantisshrimp.imports import *
from fastai2.vision.all import *
from fastai2.metrics import Metric as FastaiMetric
from mantisshrimp import *
from mantisshrimp.hub.pennfundan import *


class FastaiMetricAdapter(FastaiMetric):
    def __init__(self, metric: Metric):
        self.metric = metric

    def reset(self):
        self.metric.reset()

    def accumulate(self, learn: Learner):
        self.metric.accumulate(learn.xb, learn.yb, learn.pred)

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
###
metric = COCOMetric(valid_records, bbox=True, mask=True)
metric = FastaiMetricAdapter(metric)

# DataLoaders
class FastaiRCNNDataloader(TfmdDL):
    def create_batch(self, b):
        return (MantisMaskRCNN.collate_fn, zip_convert)[self.prebatched](b)


train_dataloader = FastaiRCNNDataloader(train_dataset, bs=2)
valid_dataloader = FastaiRCNNDataloader(valid_dataset, bs=2)
dataloaders = DataLoaders(train_dataloader, valid_dataloader).to(torch.device("cuda"))

batch = first(train_dataloader)
# Leaner
# def adapt_loss_to_fastai(learner):
#     def _fastai_loss(preds, *yb):
#         # TODO: first argument needs to be entire batch
#         set_trace()
#         return learner.model.loss(*yb, preds)


# def mock_loss(*args, **kwargs):
#     return tensor(0.0)

# class FastaiModelAdapterCallback(Callback):
#     def after_loss(self):
#         batch = (self.learn.xb, self.learn.yb)
#         self.learn.loss = self.model.loss(batch, self.learn.pred)


class RCNNCallback(Callback):
    def begin_batch(self):
        assert len(self.xb) == len(self.yb) == 1, "Only works for single input-output"
        self.learn.xb = (self.xb[0], self.yb[0])
        self.learn.yb = ()

    def after_pred(self):
        self.learn.yb = [self.learn.xb[1]]
        self.learn.xb = [self.learn.xb[0]]

    def begin_validation(self):
        # put model in training mode so we can calculate losses for validation
        self.model.train()

    def after_loss(self):
        if not self.training:
            self.model.eval()
            self.learn.pred = self.model(*self.xb)
            self.model.train()


# def adapted_fastai_learner(dls, model, **kwargs):
#     learn = Learner..
#
# def rcnn_learner():
#     learn = adapted_fastai_learner(...)


splitter = lambda model: model.params_splits()

cbs = [RCNNCallback()]
learn = Learner(
    dls=dataloaders,
    model=model,
    loss_func=MantisMaskRCNN.loss,
    cbs=cbs,
    metrics=[metric],
    splitter=splitter,
)
# patches
class RCNNAvgLoss(AvgLoss):
    def accumulate(self, learn):
        bs = len(learn.yb)
        self.total += to_detach(learn.loss.mean()) * bs
        self.count += bs


recorder = [cb for cb in learn.cbs if isinstance(cb, Recorder)][0]
recorder.loss = RCNNAvgLoss()

learn.fit(1, lr=2e-4, cbs=[])
