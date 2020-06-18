from mantisshrimp import *
from mantisshrimp.imports import *
from mantisshrimp.hub.pennfundan import *
import albumentations as A

source = get_pennfundan_data()
parser = PennFundanParser(source)

splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)

train_transform = AlbuTransform([A.HorizontalFlip()])
train_dataset = Dataset(train_records)
valid_dataset = Dataset(valid_records)

train_dataloader = MantisMaskRCNN.dataloader(train_dataset, batch_size=2)
valid_dataloader = MantisMaskRCNN.dataloader(valid_dataset, batch_size=2)


class PersonModel(MantisMaskRCNN):
    def configure_optimizers(self):
        opt = SGD(self.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-4)
        return opt


model = PersonModel(2)

trainer = Trainer(max_epochs=3, gpus=1)

trainer.fit(model, train_dataloader, valid_dataloader)
