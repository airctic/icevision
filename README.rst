MantisShrimp
============

   MantisShrimp is in very early development, all contributions are
   welcome! Be sure to check the ``issues`` board =)

|tests| |docs| |codecov| |black| |license|

--------------

``MantisShrimp`` provides a unified data API for object detection that can be used with any library (it can even be injected in code that was not supposed to work as a library, take a look at ``hub/detr`` for an example).

Install
-------

Install pytorch via your preferred way.

.. code:: python

   pip install -r requirements.txt
   pip install git+git://github.com/lgvaz/mantisshrimp.git

Quick start
-----------

Check `this`_ tutorial file for a quick introduction.

Be sure to also check all other tutorials in the ``tutorials/`` folder.

Contributing
------------
Check our `contributing guide`_.

FAQs and Feature Requests
--------------------------

Please check our `FAQs`_ page. For Feature Requests and more questions raise a github `issue`_.

We will be happy to answer.

.. _this: https://lgvaz.github.io/mantisshrimp/tutorials/wheat.html
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

