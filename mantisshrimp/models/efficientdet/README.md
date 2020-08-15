# EffecientDet Model

[**Source**](https://github.com/airctic/mantisshrimp/tree/master/mantisshrimp/models/efficientdet)

EffecientDet is one of the effecient and fastest object detection model that also uses more constrained resources in comparison to other models (Fig. 1). It was introduced in the following paper: [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)


## Usage
To train an EffecientDet model, you need to call the following highlighted functions shown here below. 

### **Common part to both Fastai and Pytorch-Lightning Training Loop**

- **DataLoaders:** Manstisshrimp creates both the train DataLoader and the valid DataLoader for both the fastai Learner and the Pytorch-Lightning Trainer  

``` python hl_lines="2 3"
# DataLoaders
train_dl = efficientdet.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
valid_dl = efficientdet.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)
```

### **Fastai Example**
There are 2 locations where the EffecientDet functions are called:

- **Model:** Mantisshrimp creates an EffecientDet model implemented by [Ross Wightman](https://github.com/rwightman/efficientdet-pytorch). The model accepts a variety of backbones. In following example, the **tf_efficientdet_lite0** is chosen. We can also chose one of the **efficientdet_d0** to **efficientdet_d7** backbones, or any other backbones listed here below

- **Fastai Learner:** It glues the EffecientDet model with the EffecientDet DataLoaders as well as the metrics and any other fastai Learner arguments.

``` python hl_lines="2 8"
# Model
model = efficientdet.model(
    model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size
)

# Fastai Learner
metrics = [COCOMetric()]
learn = efficientdet.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)
```

### **Pytorch-Lightning Example**
The Pytorch-Lightning example is quiet similar to the fastai one in a sense it uses the same DataLoaders objects, and it creates an EffecientDet model that is passed to the Pytorch-Lightning Trainer as follow:

- **Model:** Mantisshrimp creates an EffecientDet model implemented by [Ross Wightman](https://github.com/rwightman/efficientdet-pytorch). 

- **Pytorch-Lightning Trainer:** It glues the EffecientDet model with the EffecientDet DataLoaders. In Pytorch-Lightning, the metrics are passed to the model as opposed to fastai where it is passed to the Learner.

``` python hl_lines="2 6 9"
# Train using pytorch-lightning
class LightModel(efficientdet.lightning.ModelAdapter):
    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1e-4)

light_model = LightModel(model, metrics=metrics)

trainer = pl.Trainer(max_epochs=10, gpus=1)
trainer.fit(light_model, train_dl, valid_dl)
```

## Background: Paper Abstract
Model efficiency has become increasingly important in computer vision. In this paper, we systematically study neural network architecture design choices for object detection and propose several key optimizations to improve efficiency.
First, we propose a weighted bi-directional feature pyramid network (BiFPN), which allows easy and fast multiscale feature fusion; Second, we propose a compound scaling method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class prediction networks at the same time. 

Based on these optimizations and better backbones, we have developed a new family of object detectors, called EfficientDet, which consistently achieve much better efficiency than prior art across a wide spectrum of resource constraints. In particular, with singlemodel and single-scale, our EfficientDet-D7 achieves stateof-the-art 55.1 AP on COCO test-dev with 77M parameters and 410B FLOPs1, being 4x – 9x smaller and using 13x – 42x fewer FLOPs than previous detectors. 

![image](https://airctic.github.io/mantisshrimp/images/effecientdet-fig1.png)

![image](https://airctic.github.io/mantisshrimp/images/effecientdet-fig2.png)

![image](https://airctic.github.io/mantisshrimp/images/effecientdet-fig3.png)

## Mantisshrimp Quick Example: How to train the **PETS Dataset**

[**Source Code**](https://airctic.github.io/mantisshrimp/examples/efficientdet_pets_exp/)
![image](https://airctic.github.io/mantisshrimp/images/effecientdet-training.png)


## References

### Paper
[EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)


### Implementation

We use Ross Wightman's implementation which is an accurate port of the official TensorFlow (TF) implementation that accurately preserves the TF training weights

[EfficientDet (PyTorch)](https://github.com/rwightman/efficientdet-pytorch)

Any backbone in the timm model collection that supports feature extraction (features_only arg) can be used as a bacbkone.
Currently this  includes  all models implemented by the EficientNet and MobileNetv3 classes (which also includes MNasNet, MobileNetV2, MixNet and more)


