Contributing to MantisShrimp
============================

   MantisShrimp is in very early development, all contributions are
   welcome! Be sure to check the ``issues`` board =)

|tests| |docs| |codecov| |black| |license|

--------------

Built on top of `pytorch-lightining`_, ``MantisShrimp`` is an object
detection framework focused on application

Follow these steps
------------------

1. Fork the repo to your own github account. click the Fork button to create your own repo copy under your GitHub account.
Once forked, you're responsible for keeping your repo copy up-to-date with the upstream mantisshrimp repo.

2. Download a copy of your remote username/mantisshrimp repo to your local machine. 
This is the working directory where you will make changes:

.. code:: bash
    
    $ git clone https://github.com/username/mantisshrimp.git

3. Install the requirments. You many use miniconda or conda as well.

.. code:: bash
    
    $ pip install -r requirments.txt

4. Set the upstream to sync with this repo. This will keep you in sync with mantisshrimp easily. 

.. code:: bash
    
    $ git remote add upstream https://github.com/lgvaz/mantisshrimp.git

Updating
--------
1. Pull the upstream repo i.e. this repo.

.. code:: bash
    
    $ git checkout master
    $ git pull upstream master

Creating PR
-----------

1. After you update the repo from mantisshrimp/master, create a new branch from the local master branch.

.. code:: bash

   $ git checkout -b feature-name
   $ git branch
    master 
    * feature_name

2. Make changes. Edit files in your favorite editor and format the code with black.

3. Commit your file change

.. code:: bash

    # View changes
    git status  # See which files have changed
    git diff    # See changes within files

    git add path/to/file.md
    git commit -m "Your meaningful commit message for the change."

Add more commits, if necessary.

4. Create a pull request

Upload your local branch to your remote GitHub repo (github.com/username/mantisshrimp)

.. code:: bash

    git push

After the push completes, a message may display a URL to automatically submit a pull request to the upstream repo. 
If not, go to the mantisshrimp main repo and GitHub will prompt you to create a pull request.

5. Review

Maintainers and other contributors will review your pull request. 
Please participate in the discussion and make the requested changes.
When your pull request is approved, it will be merged into the upstream mantisshrimp repo.

**Note: -**
MantisShrimp has CI checking. It will automatically check your code for build as well.


.. _pytorch-lightining: https://github.com/PyTorchLightning/pytorch-lightning
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
