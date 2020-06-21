MantisShrimp
============

   MantisShrimp is in very early development, all contributions are
   welcome! Be sure to check the ``issues`` board =)

|tests| |docs| |codecov| |black| |license|

--------------

image:: https://github.com/lgvaz/mantisshrimp/images/mantisshrimp-logo.png

Why Mantishrimp
---------------
- Mantisshrimp: An object-detection library
- Built on top of different libraries/framework such as Pytorch Lightning and Pytorch
- Features a Unified Data API such: common Parsers (COCO, etc.),
- User-defined Parsers (e.g. WheatParser)
- Provides flexible model implementations using different backbones
- Helps both researchers and DL engineers in reproducing, replicating published models
- Facilitates applying both existing and new models to standard datasets as well as custom datasets


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

Install
-------

Install pytorch via your preferred way.

.. code:: bash

   pip install git+git://github.com/lgvaz/mantisshrimp.git
   pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

Quick start
-----------

Examples: Notebooks
-------------------
`wheat`_ tutorial: shows how to create a custom parser (WheatParser), and train the **wheat dataset**.


Be sure to also check all other tutorials in the ``tutorials/`` folder.

Contributing
------------
Check our `contributing guide`_.

FAQs and Feature Requests
--------------------------

Please check our `FAQs`_ page. For Feature Requests and more questions raise a github `issue`_.

We will be happy to answer.

.. _wheat: https://lgvaz.github.io/mantisshrimp/tutorials/wheat.html
.. _contributing guide: https://lgvaz.github.io/mantisshrimp/contributing.html
.. _FAQs: https://lgvaz.github.io/mantisshrimp/faqs.html
.. _issue: https://github.com/lgvaz/mantisshrimp/issues/

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

