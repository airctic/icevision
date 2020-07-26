## A- Local Installation using pypi

Using pypi repository, you can install mantisshrimp and its
dependencies:

Install PyTorch as per your preference from
[here](https://pytorch.org/get-started/locally/#start-locally).

Installing fastai and/or Pytorch-Lightning packages

```bash
pip install fastai2
pip install pytorch-lightning
```

Installing albumentations package

```bash
pip install albumentations
```

Installing mantisshrimp package using its github repo

```bash
pip install git+git://github.com/lgvaz/mantisshrimp.git[all]
```

## B- Local Installation using conda

Use the following command in order to create a conda environment called
**mantis** (the name is set in the environment.yml file)

```bash
conda env create -f environment.yml
```

### Activating mantis conda environment

To activate the newly created mantis virtual environment, run the
following command:

```bash
conda activate mantis
```

### C- Common step: cocoapi Installation: for both pypi and conda installation

#### C.1- Installing **cocoapi** in Linux:

```bash
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
```

#### C.2- Installing **cocoapi** in Windows:

pycoco cannot be installed using the command above (see
[issue-185](https://github.com/cocodataset/cocoapi/issues/185) in the
cocoapi repository). We are using this workaround:

```bash
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
```

### D- Updating mantis conda environment

To update mantis conda environment, all you need to do is update the
content of your environment.yml file accordingly and then run the
following command:

```bash
conda env update -f environment.yml  --prune
```
```
