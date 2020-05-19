# MantisShrimp
> MantisShrimp is in very early development, all contributions are welcome! Be sure to check the `issues` board =)


Built on top of [pytorch-lightining](https://github.com/PyTorchLightning/pytorch-lightning), `MantisShrimp` is an object detection framework focused on application  

Mantissrhimp provides a `DataParser` interface that simplifies the time consuming task of getting the data ready for the model, a `Tfm` interface that makes it real easy to add any transforms library to the data pipeline, and a mid and a high level interface for training the model.

## Install

```python
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install git+git://github.com/lgvaz/mantisshrimp.git
```

## Quick start

Check [this](https://github.com/lgvaz/mantisshrimp/blob/master/examples/wheat.ipynb) tutorial notebook for a quick introduction.  

Be sure to also check other examples in the `examples/` folder.
