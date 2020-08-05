!!! danger "Important"  
  We currently only support Linux/MacOS installations


### Option 1: Installing from pypi repository **[Coming Soon!]**
 
#### All Packages
To install mantisshrimp package together with all dependencies:

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
!pip install git+git://github.com/airctic/mantisshrimp.git#egg=mantisshrimp[all]
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

> **Note**: You can check out the following blog post: [3 ways to pip install a package ](https://ai-fast-track.github.io/blog/python/2020/03/17/how-to-pip-install-package.html) for more a detailed explantion on how to choose the most convenient option 
for you. 

