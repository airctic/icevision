from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.hub.pennfundan import *
from mantisshrimp.engines.fastai import *

### Unified setup ###
source = get_pennfundan_data()
parser = PennFundanParser(source)
splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)
train_dataset = Dataset(train_records)
valid_dataset = Dataset(valid_records)
model = MantisMaskRCNN(num_classes=2)
train_dataloader = model.dataloader(train_dataset, batch_size=2, num_workers=2)
valid_dataloader = model.dataloader(valid_dataset, batch_size=2, num_workers=2)
metric = COCOMetric(valid_records, bbox=True, mask=True)
###

### fastai ###
learn = rcnn_learner(
    dls=[train_dataloader, valid_dataloader], model=model, metrics=[metric]
)

learn.fine_tune(3, lr=2e-4)
###

### lightning ###
class LightningModelAdapter(LightningModule, ABC):
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
        self.reset_metrics()

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def accumulate_metrics(self, xb, yb, preds):
        for metric in self.metrics:
            metric.accumulate(xb, yb, preds)

    def finalize_metrics(self) -> dict:
        logs = {}
        for metric in self.metrics:
            value = metric.finalize()
            logs[metric.name] = value
        return logs


class RCNNLightningAdapter(LightningModelAdapter, ABC):
    def __init__(self, model: MantisRCNN, metrics: List[Metric]):
        super().__init__(metrics=metrics)
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        preds = self(xb, yb)
        loss = self.model.loss(preds, yb)
        log = {"train/loss": loss}
        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        with torch.no_grad():
            self.train()
            preds = self(xb, yb)
            loss = self.model.loss(preds, yb)
            self.eval()
            preds = self(xb)
            self.accumulate_metrics(xb, yb, preds)
        return {"valid/loss": loss}

    def validation_epoch_end(self, outs):
        loss_log = {k: torch.stack(v).mean() for k, v in mergeds(outs).items()}
        metrics_log = self.finalize_metrics()
        log = {**loss_log, **metrics_log}
        return {"val_loss": log["valid/loss"], "log": log}


# TODO: Default support for metrics?
class LightModel(RCNNLightningAdapter):
    def configure_optimizers(self):
        opt = SGD(self.parameters(), 2e-4, momentum=0.9)
        return opt


light_model = LightModel(model)
trainer = Trainer(max_epochs=3, gpus=1)
trainer.fit(light_model, train_dataloader, valid_dataloader)
###
