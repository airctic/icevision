from mantisshrimp import *
from mantisshrimp.imports import *
from mantisshrimp.hub.pennfundan import *
from mantisshrimp.hub.detr import *
import albumentations as A

source = get_pennfundan_data()
parser = PennFundanParser(source)

splitter = RandomSplitter([0.8, 0.2])
train_records, valid_records = parser.parse(splitter)

# for maskrcnn
train_transform = AlbuTransform([A.HorizontalFlip()])

train_dataset = Dataset(train_records, train_transform)
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

# for detr
train_transform = detr_transform("train")
valid_transform = detr_transform("val")

train_dataset = DetrDataset(train_records, train_transform)
valid_dataset = DetrDataset(valid_records, valid_transform)

args_parser = detr_args_parser()
args = args_parser.parse_args("")
args.fine_tune = True
args.num_classes = 2

run_detr(args=args, dataset_train=train_dataset, dataset_val=valid_dataset)
