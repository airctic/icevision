from pathlib import Path
import shutil
import os
import sys


def convert_repo_to_fork(path2repo):

    path_fork = path2repo.parent / "yolov5-icevision"  # path of our fork

    if path_fork.is_dir():
        shutil.rmtree(path_fork)
    path_fork.mkdir(exist_ok=True)

    _ = shutil.copytree(path2repo, path_fork, dirs_exist_ok=True)

    (path_fork / "yolov5").mkdir(exist_ok=True)

    sources = [
        path_fork / d
        for d in next(os.walk(path_fork))[1]
        if "git" not in d and "yolo" not in d
    ]
    dests = [(path_fork / "yolov5") / source.stem for source in sources]

    for source, dest in zip(sources, dests):
        _ = shutil.move(source, dest)

    init_content = """
__version__ = "4.0.0"

import imp
_, path, _ = imp.find_module('yolov5')
    """
    f = open(path_fork / "yolov5" / "__init__.py", "w+")
    f.write(init_content)
    f.close()

    manifest_content = """
 # Include the README
 include *.md
 # Include the license file
 include LICENSE
 # Include setup.py
 include setup.py
 # Include the data files
 recursive-include yolov5/data *
 recursive-include yolov5/models *
    """
    f = open(path_fork / "MANIFEST.in", "w+")
    f.write(manifest_content)
    f.close()

    pypi_release_content = """
name: PyPi Release
on:
release:
    types: [created]
workflow_dispatch:
jobs:
deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
    uses: actions/setup-python@v2
    with:
        python-version: '3.x'
    - name: Setup dependencies
    run: python -m pip install --user --upgrade setuptools wheel
    - name: Build package
    run: python setup.py sdist bdist_wheel
    - name: Publish package to TestPyPI
    uses: pypa/gh-action-pypi-publish@master
    with:
        user: __token__
        password: ${{ secrets.test_pypi_password }}
        repository_url: https://test.pypi.org/legacy/
        verbose: true
    - name: Publish package to PyPI
    uses: pypa/gh-action-pypi-publish@master
    with:
        user: __token__
        password: ${{ secrets.pypi_password }}
        verbose: true
    """
    f = open(path_fork / ".github/workflows/pypi-release.yml", "w+")
    f.write(pypi_release_content)
    f.close()

    setup_cfg_content = """
# reference: https://setuptools.readthedocs.io/en/latest/userguide/declarative_config.html
[metadata]
name = yolov5-icevision
version = 4.0.0
author = ultralytics
author_email = glenn.jocher@ultralytics.com
description = YOLOv5
long_description = file: README.md
long_description_content_type = text/markdown
keywords = object detection, machine learning
license = GNU General Public License v3.0
classifiers =
Development Status :: 4 - Beta
Intended Audience :: Developers
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Image Recognition
[options]
python_requires = >=3.6,<4
zip_safe = False
include_package_data = True
packages = find:
install_requires = 
    Cython
    matplotlib>=3.2.2
    numpy>=1.18.5
    opencv-python>=4.1.2
    Pillow
    PyYAML>=5.3.1
    scipy>=1.4.1
    tensorboard>=2.2
    torch>=1.7.0
    torchvision>=0.8.1
    tqdm>=4.41.0
    seaborn>=0.11.0

[options.extras_require]
all =
    pandas
"""
    f = open(path_fork / "setup.cfg", "w+")
    f.write(setup_cfg_content)
    f.close()

    setup_py_content = """
from setuptools import setup

if __name__ == "__main__":
    setup()
    """
    f = open(path_fork / "setup.py", "w+")
    f.write(setup_py_content)
    f.close()

    def replace_imports(path):
        bads = ["from utils", "from models", "import utils", "import models"]
        goods = [
            "from yolov5.utils",
            "from yolov5.models",
            "import yolov5.utils",
            "import yolov5.models",
        ]

        with open(path, "r") as file:
            filedata = file.read()
            if any(bad in filedata for bad in bads):
                for bad, good in zip(bads, goods):
                    filedata = filedata.replace(bad, good)

                with open(path, "w") as file:
                    file.write(filedata)

    for root, _, files in os.walk(path_fork):
        for f in files:
            if f.endswith(".py"):
                replace_imports(os.path.join(root, f))

    with open(path_fork / ".gitignore", "r+") as f:
        d = f.readlines()
        f.seek(0)
        for i in d:
            if i.strip() != "*.cfg":
                f.write(i)
        f.truncate()


if __name__ == "__main__":
    path2repo = Path(sys.argv[1])
    convert_repo_to_fork(path2repo)
    print(
        f"Conversion completed successfully. Please find the fork here: {str(path2repo.parent / 'yolov5-icevision')}"
    )
