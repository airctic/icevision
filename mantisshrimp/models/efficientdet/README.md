# EffecientDet Model

[**Source**](https://github.com/airctic/mantisshrimp/tree/master/mantisshrimp/models/efficientdet)

EffecientDet is one of the effecient and fastest object detection model that also uses more constrained resources in comparison to other models (Fig. 1). It was introduced in the following paper:

[EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)


## Paper Abstract
Model efficiency has become increasingly important in computer vision. In this paper, we systematically study neural network architecture design choices for object detection and propose several key optimizations to improve efficiency.
First, we propose a weighted bi-directional feature pyramid network (BiFPN), which allows easy and fast multiscale feature fusion; Second, we propose a compound scaling method that uniformly scales the resolution, depth, and width for all backbone, feature network, and box/class prediction networks at the same time. 

Based on these optimizations and better backbones, we have developed a new family of object detectors, called EfficientDet, which consistently achieve much better efficiency than prior art across a wide spectrum of resource constraints. In particular, with singlemodel and single-scale, our EfficientDet-D7 achieves stateof-the-art 55.1 AP on COCO test-dev with 77M parameters and 410B FLOPs1, being 4x – 9x smaller and using 13x – 42x fewer FLOPs than previous detectors. 

![image](https://airctic.github.io/mantisshrimp/images/effecientdet-fig1.png)

![image](https://airctic.github.io/mantisshrimp/images/effecientdet-fig2.png)

![image](https://airctic.github.io/mantisshrimp/images/effecientdet-fig3.png)

## Mantisshrimp Quick Example: How to train the **Fridge Object Dataset**

[**Source Code**](https://airctic.github.io/mantisshrimp/examples/how_train_dataset_exp/)
![image](https://airctic.github.io/mantisshrimp/images/effecientdet-training.png)


## References

### Paper
[EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)


### Implementation

We use Ross Wightman's implementation which is an accurate port of the official TensorFlow (TF) implementation that accurately preserves the TF training weights

[EfficientDet (PyTorch)](https://github.com/rwightman/efficientdet-pytorch)

Any backbone in the timm model collection that supports feature extraction (features_only arg) can be used as a bacbkone.
Currently this  includes  all models implemented by the EficientNet and MobileNetv3 classes (which also includes MNasNet, MobileNetV2, MixNet and more)


