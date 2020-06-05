# MantisShrimp
> MantisShrimp is in very early development, all contributions are welcome! Be sure to check the `issues` board =)

[![CI testing](https://github.com/lgvaz/mantisshrimp/workflows/CI%20testing/badge.svg?event=push)](https://github.com/lgvaz/mantisshrimp/actions?query=workflow%3A%22CI+testing%22)
[![codecov](https://codecov.io/gh/lgvaz/mantisshrimp/branch/master/graph/badge.svg)](https://codecov.io/gh/lgvaz/mantisshrimp)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/lgvaz/mantisshrimp/blob/master/LICENSE)

---
Built on top of [pytorch-lightining](https://github.com/PyTorchLightning/pytorch-lightning), `MantisShrimp` is an object detection framework focused on application  

Mantisshrimp provides a `DataParser` interface that simplifies the time consuming task of getting the data ready for the model, a `Tfm` interface that makes it real easy to add any transforms library to the data pipeline, and a mid and a high level interface for training the model.


## Install

```python
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+git://github.com/lgvaz/mantisshrimp.git
```

## Quick start

Check [this](https://github.com/lgvaz/mantisshrimp/blob/master/examples/wheat.py) tutorial file for a quick introduction.  

Be sure to also check other examples in the `examples/` folder.
