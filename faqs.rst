FAQs
=======================

Why Mantisshrimp ?
------------------

Mantisshrimp is written in Pytorch with Pytorch lightning.

It provides flexible usage of object detection models.

Mantisshrimp plans to add support for detectron2 models as well as Pytorch hub models.

Who is Mantisshrimp for ?
-------------------------

It is for both researchers as well as developers. 

Mantisshrimp does not hide the Pytorch code which provides you the flexibility to build your object detection models.

Does it allow both fine-tuning and training from scratch ?
----------------------------------------------------------

Yes, you can do both. 

How does it make object detection tasks faster ?
-------------------------------------------------------------------------

Mantisshrimp provides datasets and parsers which make it easier to feed into object detection models.

Also it has support for `albumentations`_ and torchvision transforms for image augmentation.

It provides multiple models implemented so you can directly experiment with them, fine tune as well as train from scratch.

With  `pytorch-lightining`_ under the hood you can train on multiple GPUs, TPUs and use the torch models directly.

How do I use Mantisshrimp ?
---------------------------

Check out our tutorials, where it shows you how to perform end to end object detection task.

How do I Contribute ?
---------------------

Please have a look at `contributing guide`_.

How do I get started ?
----------------------

Please check our `docs`_ , they would provide you a detailed guide to use Mantisshrimp.

.. _albumentations: https://github.com/albumentations-team/albumentations
.. _pytorch-lightining: https://github.com/PyTorchLightning/pytorch-lightning
.. _contributing guide: https://github.com/lgvaz/mantisshrimp/blob/master/contributing.rst
.. _docs: https://lgvaz.github.io/mantisshrimp/