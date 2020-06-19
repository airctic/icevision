from mantisshrimp.imports import *
from mantisshrimp import *
from mantisshrimp.hub.pennfundan import *


class LightningModelAdapter(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        preds = self.model(*batch)
        loss = self.model.get_loss(batch, preds)
        logs = self.model.get_logs(batch, preds)
        logs = {f"train/{k}": v for k, v in logs.items()}
        return {"loss": loss, "log": logs}


# Unified setup
source = get_pennfundan_data()
parser = PennFundanParser(source)
splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)
train_dataset = Dataset(train_records)
valid_dataset = Dataset(valid_records)
model = MantisMaskRCNN(num_classes=2)
# Should also be unified eventually
train_dataloader = MantisMaskRCNN.dataloader(train_dataset)
valid_dataloader = MantisMaskRCNN.dataloader(valid_dataset)

# Not unified
class MyModel(LightningModelAdapter):
    def configure_optimizers(self):
        return Adam(self.parameters())


model = MantisMaskRCNN(2)
model = MyModel(model)

trainer = Trainer(max_epochs=2, gpus=1)
trainer.fit(model, train_dataloader)
