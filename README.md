<div align="center">
  <img src="images/icevision-logo-slogan.png" alt="logo" width="400px" style="display: block; margin-left: auto; margin-right: auto"/>
  <h2><b>An Agnostic Object Detection Framework</b></h2>
</div>

* * * * *
>**Note: "We Need Your Help"**
    If you find this work useful, please let other people know by **starring** it,
    and sharing it. 
    Thank you!
    
<div align="center">
    
[![tests](https://github.com/airctic/icevision/workflows/tests/badge.svg?event=push)](https://github.com/airctic/icevision/actions?query=workflow%3Atests)
[![docs](https://github.com/airctic/icevision/workflows/docs/badge.svg)](https://airctic.github.io/icevision/index.html)
[![codecov](https://codecov.io/gh/airctic/icevision/branch/master/graph/badge.svg)](https://codecov.io/gh/airctic/icevision)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/airctic/icevision/blob/master/LICENSE)  

[![Join Users Forum](https://withspectrum.github.io/badge/badge.svg)](https://spectrum.chat/mantis)

</div>


* * * * *

![image](images/icevision-end-to-end-training.gif)

<!-- Not included in docs - start -->
## **Contributors**

[![](https://sourcerer.io/fame/lgvaz/airctic/icevision/images/0)](https://sourcerer.io/fame/lgvaz/airctic/icevision/links/0)[![](https://sourcerer.io/fame/lgvaz/airctic/icevision/images/1)](https://sourcerer.io/fame/lgvaz/airctic/icevision/links/1)[![](https://sourcerer.io/fame/lgvaz/airctic/icevision/images/2)](https://sourcerer.io/fame/lgvaz/airctic/icevision/links/2)[![](https://sourcerer.io/fame/lgvaz/airctic/icevision/images/3)](https://sourcerer.io/fame/lgvaz/airctic/icevision/links/3)[![](https://sourcerer.io/fame/lgvaz/airctic/icevision/images/4)](https://sourcerer.io/fame/lgvaz/airctic/icevision/links/4)[![](https://sourcerer.io/fame/lgvaz/airctic/icevision/images/5)](https://sourcerer.io/fame/lgvaz/airctic/icevision/links/5)[![](https://sourcerer.io/fame/lgvaz/airctic/icevision/images/6)](https://sourcerer.io/fame/lgvaz/airctic/icevision/links/6)[![](https://sourcerer.io/fame/lgvaz/airctic/icevision/images/7)](https://sourcerer.io/fame/lgvaz/airctic/icevision/links/7)

## Installation

```bash
pip install icevision[all]
pip install pycocotools@https://github.com/lgvaz/cocoapi/archive/master.zip#subdirectory=PythonAPI&egg=pycocotools-2.0
pip install omegaconf effdet@https://github.com/rwightman/efficientdet-pytorch/archive/master.zip#egg=effdet-0.1.4
```

For more installation options, check our [docs](https://airctic.github.io/icevision/install/).

**Important:** We currently only support Linux/MacOS.
<!-- Not included in docs - end -->


## Quick Example: How to train the **PETS Dataset**
[**Source Code**](https://airctic.github.io/icevision/examples/training/)
![image](images/icevision-readme.png)



## The Problem We Are Solving

-   Object dectection datasets come in different sizes and most
    impotantly have different annotations formats ranging from the
    stanndard formarts such COCO and VOC to more self-tailored formats
-   When new object detection models are released with some source code,
    the latter is very often written in non-portable way: The source
    code is difficult to use for other datasets because of some
    hard-coded parts coupled with self developed tweaks
-   Both researchers and DL coders have to deploy a lot of effort to use
    many SOTA models for their own use-cases and/or to craft an enhanced
    model based on those already published

## Our Solution

IceVision library provides some elegant solutions in those 2
fundamental components:

**1- A Unified Data API**

Out of the box, we offer several annotation parsers that translates
different annotation formats into a very flexibe parser:

* By default, we offer differents standard format parsers such as COCO
  and VOC.
* We host a community curated parsers where community contributors
  publish their own parsers to be shared, and therefore save time and
  energy in creating similar parsers over and over.
* We provide some intuitive tutorials that walk you through the steps
  of creating your own parser. Please, consider sharing it with the
  whole community.

**2- A Universal Adapter to different DL Libraries**

* IceVision provides a universal adapter that allows you to hook up
  your dataset to the DL library of your choice (fastai, Pytorch
  Lightning and Pytorch), and train your model using a familiar API.
* Our library allows you to choose one of the public implementations
  of a given model, plug it in icevision model adapter, and
  seamlessly train your model.
* As a bonus, our library even allows to experiment with another DL
  library. Our tutorials have several examples showing you how to
  train a given model using both fastai and Pytorch Lightning
  libraries side by side.


## Happy Learning!
If you need any assistance, feel free to:

[Join our Users Forum](https://spectrum.chat/mantis)

[Join our Devs Forum](https://discord.gg/QxHctJF)
