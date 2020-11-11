# Faster RCNN Model

[**Source**](https://github.com/airctic/icevision/tree/master/icevision/models/rcnn/faster_rcnn/)

Faster RCNN is one of the most popular object detection model. It was introduced in the following paper:
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

## Usage
To train a Faster RCNN model, you need to call the following highlighted functions shown here below. 

### **Common part to both Fastai and Pytorch-Lightning Training Loop**

- **DataLoaders:** Manstisshrimp creates common train DataLoader and valid DataLoader to both fastai Learner and Pytorch-Lightning Trainer   

``` python hl_lines="2 3"
# DataLoaders
train_dl = faster_rcnn.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)
```


- **Model:** IceVision creates a Faster RCNN model implemented in [torchvision FasterRCNN](https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py). The model accepts a variety of backbones. In following example, we use the default [fasterrcnn_resnet50_fpn](https://github.com/pytorch/vision/blob/27278ec8887a511bd7d6f1202d50b0da7537fc3d/torchvision/models/detection/faster_rcnn.py#L291) model. We can also choose one of the following [backbones](https://github.com/airctic/icevision/blob/master/icevision/backbones/resnet_fpn.py): resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2


``` python hl_lines="2"
# Model
model = faster_rcnn.model(num_classes=len(class_map))
```

- **How to use a different backbone:** "resnet18" example
``` python hl_lines="2 4"
# Backbone
backbone = faster_rcnn.backbones.resnet_fpn.resnet18(pretrained=True)
# Model
model = faster_rcnn.model(backbone=backbone, num_classes=len(class_map))
```


### **Fastai Example**
Once the DataLoaders and the Faster RCNN model are created, we create the fastai Learner. The latter uses the DataLoaders and the Faster RCNN model shared with the Pytorch-Lightning Trainer (as shown in the Pytorch-Lightning example here below):

- **Fastai Learner:** It glues the Faster RCNN model with the DataLoaders as well as the metrics and any other fastai Learner arguments. In the code snippet shown here below, we highlight the parts related to the Faster RCNN model.

``` python hl_lines="3-5"
# Fastai Learner
metrics = [COCOMetric()]
learn = faster_rcnn.fastai.learner(
    dls=[train_dl, valid_dl], model=model, metrics=metrics
)
learn.fine_tune(10, 1e-2, freeze_epochs=1)
```

### **Pytorch-Lightning Example**
The Pytorch-Lightning example is quiet similar to the fastai one in a sense it uses the same DataLoaders objects, and the same Faster RCNN model. Those are subsenquently passed on to the Pytorch-Lightning Trainer:

- **Pytorch-Lightning Trainer:** It glues the Faster RCNN model with the DataLoaders. In Pytorch-Lightning, the metrics are passed to the model object as opposed to fastai where it is passed to the Learner object. In the code snippet shown here below, we highlight the parts related to the Faster RCNN model.

``` python hl_lines="2 6 9"
# Train using pytorch-lightning
class LightModel(faster_rcnn.lightning.ModelAdapter):
    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-4)

light_model = LightModel(model, metrics=metrics)

trainer = pl.Trainer(max_epochs=10, gpus=1)
trainer.fit(light_model, train_dl, valid_dl)
```


## How to train the **PETS Dataset** using **Faster RCNN**

[**Source Code**](https://airctic.github.io/icevision/examples/training/)
![image](https://airctic.github.io/icevision/images/icevision.png)


## Paper Introduction
Faster R-CNN is built upon the knowledge of Fast RCNN which indeed built upon the ideas of RCNN and SPP-Net. In their paper, the authors introduced a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. 

![image](https://airctic.github.io/icevision/images/fast-rcnn-vs-faster-rcnn.png)

![image](https://airctic.github.io/icevision/images/faster-rcnn-fig-2.png)

![image](https://airctic.github.io/icevision/images/faster-rcnn-fig-3.png)


## References

### Paper
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

### Blog Posts
[An overview of deep-learning based object-detection algorithms](https://medium.com/@fractaldle/brief-overview-on-object-detection-algorithms-ec516929be93)

[Guide to build Faster RCNN in PyTorch](https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439)

### Torchvision Implementation

[Torchvision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

[Torchvision Faster RCNN Implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py)

