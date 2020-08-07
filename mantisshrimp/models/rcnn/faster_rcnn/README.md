# Faster RCNN Model

[**Source**](https://github.com/airctic/mantisshrimp/tree/master/mantisshrimp/models/rcnn/faster_rcnn/)

Faster RCNN is one of the most popular object detection model. It was introduced in the following paper:
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)


## Introduction
Faster R-CNN is built upon the knowledge of Fast RCNN which indeed built upon the ideas of RCNN and SPP-Net. In their paper, the authors introduced a Region Proposal Network (RPN) that shares full-image convolutional features with the detection network, thus enabling nearly cost-free region proposals. An RPN is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position. 

![image](https://airctic.github.io/mantisshrimp/images/fast-rcnn-vs-faster-rcnn.png)

![image](https://airctic.github.io/mantisshrimp/images/faster-rcnn-fig-2.png)

![image](https://airctic.github.io/mantisshrimp/images/faster-rcnn-fig-3.png)


## Mantisshrimp Quick Example: How to train the **PETS Dataset**

[**Source Code**](https://airctic.github.io/mantisshrimp/examples/training/)
![image](https://airctic.github.io/mantisshrimp/images/mantis-readme.png)


## References

### Paper
[Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)

### Blog Posts
[An overview of deep-learning based object-detection algorithms](https://medium.com/@fractaldle/brief-overview-on-object-detection-algorithms-ec516929be93)

[Guide to build Faster RCNN in PyTorch](https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439)

### Torchvision Implementation

[Torchvision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

[Torchvision Faster RCNN Implementation](https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py)

