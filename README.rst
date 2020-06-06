MantisShrimp
============

   MantisShrimp is in very early development, all contributions are
   welcome! Be sure to check the ``issues`` board =)

|CI testing| |docs| |codecov| |black| |license|

--------------

Built on top of `pytorch-lightining`_, ``MantisShrimp`` is an object
detection framework focused on application

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
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://github.com/lgvaz/mantisshrimp/blob/master/LICENSE
.. |docs| image:: https://github.com/lgvaz/mantisshrimp/workflows/Build%20docs/badge.svg
   :target: https://lgvaz.github.io/mantisshrimp/index.html
