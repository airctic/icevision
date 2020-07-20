FAQs
=======================

What problem are you solving?
-----------------------------

- Object dectection datasets come in different sizes and most impotantly have different annotations formats ranging from the stanndard formarts such COCO and VOC to more self-tailored formats

- When new object detection models are released with some source code, the latter is very often written in non-portable way: The source code is difficult to use for other datasets because of some hard-coded parts coupled with self developed tweaks

- Both researchers and DL coders have to deploy a lot of effort to use many SOTA models for their own use-cases and/or to craft an enhanced model based on those already published

What is your solution
---------------------
Mantisshrimp library provides some elegant solutions in those 2 fundamental components:

**1- A Unified Data API:** 

Out of the box, we offer several annotation parsers that translates different annotation formats into a very flexibe parser:

A. By default, we offer  differents standard format parsers such as COCO and ROC,

B. We host a community curated parsers where community contributors publish their own parsers to be shared, and therefore save time and energy in creating similar parsers over and over,
 
C. We provide some intuitive tutorials that walk you through the steps of creating your own parser. Please, consider sharing it with the whole community.


**2- A Universal Adapter to different DL Libraries:**

A. Mantisshrimp provides a universal adapter that allows you to hook up your dataset to the DL library of your choice (`fastai2`_, `Pytorch-Lightning`_ and `Pytorch`_), and train your model using a familiar API,

B. Our library allows you to choose one of the public implementations of a given model, plug it in mantisshrimp model adapter, and seamlessly train your model,

C. As a bonus, our library even allows to experiment with another DL library. Our tutorials have several examples showing you how to train a given model using both fastai and Pytorch Lightning libraries side by side.


Why Mantisshrimp ?
------------------

- An agnostic object-detection library
- Connects to different libraries/framework such as fastai, Pytorch Lightning, and Pytorch
- Features a Unified Data API such: common Parsers (COCO, VOC, etc.)
- Integrates community maintaned parsers for custom datasets shared on parsers hub
- Provides flexible model implementations using different backbones
- Helps both researchers and DL engineers in reproducing, replicating published models
- Facilitates applying both existing and new models to standard datasets as well as custom datasets

What Deep Learning Libraries does Mantisshrimp support
------------------------------------------------------

Out of the box, Mantisshrimp supports both `fastai2`_ and `Pytorch-Lightning`_ librairies. We plan adding support for other DL libraries in the near future


What is Mantisshrimp audience ?
-------------------------------

It is for both researchers as well as developers. 

Mantisshrimp offers a great flexibility by allowing the users to use either off-shelf models or build their own using a wide variety of architectures and backbones.

Does it allow both fine-tuning and training from scratch ?
----------------------------------------------------------

Yes, you can do both. 

How does it make object detection tasks faster ?
-------------------------------------------------------------------------

Mantisshrimp provides parsers which make it very easy to feed custom datasets into object detection models.

It works with any transform library with minimal setup, we currently support `albumentations`_ out of the box.

It provides multiple models implemented so you can directly experiment with them, fine tune as well as train from scratch.

With  `Pytorch-Lightning`_ under the hood you can train on multiple GPUs, TPUs and use the torch models directly.

How do I use Mantisshrimp ?
---------------------------

Check out our `Getting Started Guide`_, where it shows you how to perform end to end object detection task.

How do I Contribute ?
---------------------

Please have a look at `contributing guide`_.

How do I get started ?
----------------------

Please check our `docs`_ , they would provide you a detailed guide to use Mantisshrimp.

.. _albumentations: https://github.com/albumentations-team/albumentations
.. _Pytorch: https://github.com/PyTorch
.. _Pytorch-Lightning: https://github.com/PyTorchLightning/pytorch-lightning
.. _fastai2: https://github.com/fastai/fastai2
.. _contributing guide: https://lgvaz.github.io/mantisshrimp/contributing.html
.. _docs: https://lgvaz.github.io/mantisshrimp/
.. _Getting Started Guide: https://lgvaz.github.io/mantisshrimp/tutorials/getting_started.html
