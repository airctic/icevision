## A- Local Installation using pypi

There are 3 ways to install mantisshrimp and its dependencies using `pip install`. 

> **Note**: You can check out the following blog post: [3 ways to pip install a package ](https://ai-fast-track.github.io/blog/python/2020/03/17/how-to-pip-install-package.html) for more a detailed explantion on how to choose the most convenient option 
for you. 


### Option 1: Installing from pypi repository **[Coming Soon!]**
 
#### All Packages
To install mantisshrimp package and both Fastai and Pytorch-Lightning libraries, run the following command:

```bash
pip install mantisshrimp[all]
```

#### Mantisshrimp + Fastai
To install mantisshrimp package and only the Fastai library, run the following command:

```bash
pip install mantisshrimp[fastai]
```

#### Mantisshrimp + Pytorch-Lightning
To install mantisshrimp package and only the Pytorch-Lightning library, run the following command:

```bash
pip install mantisshrimp[pytorch_lightning]
```

### Option 2: Installing a non-editable package from GitHub **[Already Available]**

To install the mantisshrimp package from its GitHub repo, run the command here below. This option can be used in Google Colab,
for example, where you might install the mantisshrimp latest version (from the `master` branch)

```bash
pip install git+git://github.com/airctic/mantisshrimp.git[all]
```

### Option 3: Installing an editable package from GitHub **[Already Available]**
> **Note:** This method is used by developers who are usually either:
>
> - actively contributing to `mantisshrimp` project by adding new features or fixing bugs, or 
> - creating their own modules, and making sure that their source code stay in sync with the `mantisshrimp` latest version.

All we have to do is to follow these 3 simple steps by running the following commands:

```bash
git clone --depth=1 https://github.com/airctic/mantisshrimp.git
cd mantisshrimp
pip install .[all]
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

> **Note:**
> Once you activate the conda environment, follow the steps described, here above, in order to `pip install` 
> the mantisshrimp package and its dependencies: **A- Local Installation using pypi** 




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

