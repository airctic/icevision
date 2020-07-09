MantisShrimp
============

   MantisShrimp is in very early development, all contributions are
   welcome! Be sure to check the ``issues`` board =)

|tests| |docs| |codecov| |black| |license|

--------------

.. image:: images/mantisshrimp-logo.png

The Problem We Are Solving
--------------------------

- Object dectection datasets come in different sizes and most impotantly have different annotations formats ranging from the stanndard formarts such COCO and VOC to more self-tailored formats

- When new object detection models are released with some source code, the latter is very often written in non-portable way: The source code is difficult to use for other datasets because of some hard-coded parts coupled with self developed tweaks

- Both researchers and DL coders have to deploy a lot of effort to use many SOTA models for their own use-cases and/or to craft an enhanced model based on those already published

Our Solution
------------
Mantisshrimp library provides some elegant solutions in those 2 fundamental components:

**1- A Unified Data API:** 

Out of the box, we offer several annotation parsers that translates different annotation formats into a very flexibe parser:

A. By default, we offer  differents standard format parsers such as COCO and ROC,

B. We host a community curated parsers where community contributors publish their own parsers to be shared, and therefore save time and energy in creating similar parsers over and over,
 
C. We provide some intuitive tutorials that walk you through the steps of creating your own parser. Please, consider sharing it with the whole community.


**2- A Universal Adapter to different DL Libraries:**

A. Mantisshrimp provides a universal adapter that allows you to hook up your dataset to the DL library of your choice (fastai, Pytorch Lightning and Pytorch), and train your model using a familiar API,

B. Our library allows you to choose one of the public implementations of a given model, plug it in mantisshrimp model adapter, and seamlessly train your model,

C. As a bonus, our library even allows to experiment with another DL library. Our tutorials have several examples showing you how to train a given model using both fastai and Pytorch Lightning libraries side by side.


Why Mantishrimp
---------------
- An agnostic object-detection library
- Connects to different libraries/framework such as fastai, Pytorch Lightning, and Pytorch
- Features a Unified Data API such: common Parsers (COCO, VOC, etc.)
- Integrates community maintaned parsers for custom datasets shared on parsers hub
- Provides flexible model implementations using different backbones
- Helps both researchers and DL engineers in reproducing, replicating published models
- Facilitates applying both existing and new models to standard datasets as well as custom datasets

**Note:**  If you find this work useful, please let other people know by **starring** it. Thank you!

Hall of Fame
------------

This library is only made possible because of @all-contributors, thank you ♥️ ♥️ ♥️ 

|image0|\ |image1|\ |image2|\ |image3|\ |image4|\ |image5|\ |image6|\ |image7|

.. |image0| image:: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/images/0
   :target: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/links/0
.. |image1| image:: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/images/1
   :target: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/links/1
.. |image2| image:: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/images/2
   :target: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/links/2
.. |image3| image:: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/images/3
   :target: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/links/3
.. |image4| image:: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/images/4
   :target: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/links/4
.. |image5| image:: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/images/5
   :target: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/links/5
.. |image6| image:: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/images/6
   :target: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/links/6
.. |image7| image:: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/images/7
   :target: https://sourcerer.io/fame/lgvaz/lgvaz/mantisshrimp/links/7

A- Quick Start: Use Mantisshrimp Docker Container
-------------------------------------------------
To jumpstart using mantisshrimp package without manually installing it and its dependencies, use our docker container!

Please, follow the 3 steps:

1. Install Docker by following the instructions shown in `Docker website`_ (Only if Docker is not already installed)

2. In the terminal, pull the mantisshrimp docker image: 

.. code:: bash
   
   docker pull mantisshrimp


3. In the terminal, run the mantisshrimp docker container:  

.. code:: bash
   
   docker run -it mantisshrimp
   

Enjoy!

B- Local Installation using pypi
--------------------------------
Using pypi repository, you can install mantisshrimp and its dependencies:

Install PyTorch as per your preference from `here`_.

Installing fastai and/or Pytorch-Lightning packages

.. code:: bash

   pip install fastai2
   pip install pytorch-lightning

Installing albumentations package

.. code:: bash

   pip install albumentations

Installing mantisshrimp package using its github repo

.. code:: bash

   pip install git+git://github.com/lgvaz/mantisshrimp.git


C- Local Installation using conda
---------------------------------
Use the following command in order to create a conda environment called **mantis** (the name is set in the `environment.yml` file)

.. code:: bash

   conda env create -f environment.yml

Activating `mantis` conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To activate the newly created `mantis` virtual environment, run the following command:

.. code:: bash

   conda activate mantis


D- Common step: cocoapi Installation: for both pypi and conda installation
--------------------------------------------------------------------------


D.1- Installing **cocoapi** in Linux:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: bash

   pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
   

D.2- Installing **cocoapi** in Windows:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`pycoco` cannot be installed using the command above (see `issue-185`_ in the cocoapi repository). We are using this workaround:

.. code:: bash

   pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
   


E- Updating `mantis` conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To update `mantis` conda environment, all you need to do is update the content of your environment.yml file accordingly and then run the following command:

.. code:: bash

   conda env update -f environment.yml  --prune



Quick Example: How to train the **PETS Dataset**
-------------------------------------------------

.. code:: python

   from mantisshrimp.imports import *
   from mantisshrimp import *
   import albumentations as A

   # Load the PETS dataset
   path = datasets.pets.load()

   # split dataset lists
   data_splitter = RandomSplitter([.8, .2])

   # PETS parser: provided out-of-the-box
   parser = datasets.pets.parser(path)
   train_rs, valid_rs = parser.parse(data_splitter)

   # For convenience
   CLASSES = datasets.pets.CLASSES

   # shows images with corresponding labels and boxes
   records = train_records[:6]
   show_records(records, ncols=3, classes=CLASSES)

   # ImageNet stats
   imagenet_mean, imagenet_std = IMAGENET_STATS

   # Transform: supporting albumentations transforms out of the box
   # Transform for the train dataset
   train_tfms = AlbuTransform(
       [
           A.LongestMaxSize(384),
           A.RandomSizedBBoxSafeCrop(320, 320, p=0.3),
           A.HorizontalFlip(),
           A.ShiftScaleRotate(rotate_limit=20),
           A.RGBShift(always_apply=True),
           A.RandomBrightnessContrast(),
           A.Blur(blur_limit=(1, 3)),
           A.Normalize(mean=imagenet_mean, std=imagenet_std),
       ]
   )

   # Transform for the validation dataset
   valid_tfms = AlbuTransform(
       [
           A.LongestMaxSize(384),
           A.Normalize(mean=imagenet_mean, std=imagenet_std),
       ]
   )   

   # Create both training and validation datasets
   train_ds = Dataset(train_records, train_tfms)
   valid_ds = Dataset(valid_records, valid_tfms)

   # Create both training and validation dataloaders
   train_dl = model.dataloader(train_ds, batch_size=16, num_workers=4, shuffle=True)
   valid_dl = model.dataloader(valid_ds, batch_size=16, num_workers=4, shuffle=False)

   # Create model
   model = MantisFasterRCNN(num_classes= len(CLASSES))

   # Training the model using fastai2
   from mantisshrimp.engines.fastai import *
   learn = rcnn_learner(dls=[train_dl, valid_dl], model=model)
   learn.fine_tune(10, lr=1e-4)

   # Training the model using Pytorch-Lightning
   from mantisshrimp.engines.lightning import *
   
   class LightModel(RCNNLightningAdapter):
      def configur1e_optimizers(self):
          opt = SGD(self.parameters(), 2e-4, momentum=0.9)
          return opt

   light_model = LightModel(model)
   trainer = Trainer(max_epochs=3, gpus=1)
   trainer.fit(light_model, train_dl, valid_dl)

Contributing
------------
Check out our `contributing guide`_.

Feature Requests and questions
------------------------------

For Feature Requests and more questions raise a github `issue`_. We will be happy to assist you.  

Be sure to check the `documentation`_.  


.. _documentation: https://lgvaz.github.io/mantisshrimp/index.html
.. _contributing guide: https://lgvaz.github.io/mantisshrimp/contributing.html
.. _issue: https://github.com/lgvaz/mantisshrimp/issues/
.. _here: https://pytorch.org/get-started/locally/#start-locally
.. _issue-185: https://github.com/cocodataset/cocoapi/issues/185
.. _Docker website: https://docs.docker.com/engine/install/

.. |tests| image:: https://github.com/lgvaz/mantisshrimp/workflows/tests/badge.svg?event=push
   :target: https://github.com/lgvaz/mantisshrimp/actions?query=workflow%3Atests
.. |codecov| image:: https://codecov.io/gh/lgvaz/mantisshrimp/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/lgvaz/mantisshrimp
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://github.com/lgvaz/mantisshrimp/blob/master/LICENSE
.. |docs| image:: https://github.com/lgvaz/mantisshrimp/workflows/docs/badge.svg
   :target: https://lgvaz.github.io/mantisshrimp/index.html