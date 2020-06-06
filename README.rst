MantisShrimp
============

   MantisShrimp is in very early development, all contributions are
   welcome! Be sure to check the ``issues`` board =)

|CI testing| |codecov| |Black| |License|

--------------

Built on top of `pytorch-lightining`_, ``MantisShrimp`` is an object
detection framework focused on application

Mantisshrimp provides a ``DataParser`` interface that simplifies the
time consuming task of getting the data ready for the model, a ``Tfm``
interface that makes it real easy to add any transforms library to the
data pipeline, and a mid and a high level interface for training the
model.

Install
-------

.. code:: python

   pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
   pip install git+git://github.com/lgvaz/mantisshrimp.git

Quick start
-----------

Check `this`_ tutorial file for a quick introduction.

Be sure to also check other examples in the ``examples/`` folder.

.. _pytorch-lightining: https://github.com/PyTorchLightning/pytorch-lightning
.. _this: https://github.com/lgvaz/mantisshrimp/blob/master/examples/wheat.py

.. |CI testing| image:: https://github.com/lgvaz/mantisshrimp/workflows/CI%20testing/badge.svg?event=push
   :target: https://github.com/lgvaz/mantisshrimp/actions?query=workflow%3A%22CI+testing%22
.. |codecov| image:: https://codecov.io/gh/lgvaz/mantisshrimp/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/lgvaz/mantisshrimp
.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://github.com/lgvaz/mantisshrimp/blob/master/LICENSE