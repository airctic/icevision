!!! danger "Important"  
    We currently only support Linux/MacOS installations

!!! info "Note"  
    Please do not forget to install the other optional dependencies if you would like to use them:
    
    - MMCV+MMDetection, and/or 

    - YOLOv5 

## A- Installation using pip

### **Option 1:** Installing from pypi repository **[Stable Version]**
 
To install icevision package together with almost all dependencies:

<div class="termy">
```console
$ pip install icevision[all]
```
</div>


### **Option 2:** Installing an editable package locally **[For Developers]**

!!! info "Note"  
    This method is used by developers who are usually either:

    - actively contributing to `icevision` project by adding new features or fixing bugs, or 

    - creating their own extensions, and making sure that their source code stay in sync with the `icevision` latest version.

Then, clone the repo and install the package:
<div class="termy">
```console
$ git clone --depth=1 https://github.com/airctic/icevision.git
$ cd icevision
$ pip install -e .[all,dev]
$ pre-commit install
```
</div>


### **Option 3:** Installing a non-editable package from GitHub:

To install the icevision package from its GitHub repo, run the command here below. This option can be used in Google Colab,
for example, where you might install the icevision latest version (from the `master` branch)

<div class="termy">
```console
$ pip install git+https://github.com/airctic/icevision.git#egg=icevision[all] --upgrade
```
</div>


## B- Installation using conda
Creating a conda environment is considered as a best practice because it avoids polluting the default (base) environment, and reduces dependencies conflicts. Use the following command in order to create a conda environment called **icevision**

<div class="termy">
```console
$ conda create -n icevision python=3.8 anaconda
$ conda activate icevision
$ pip install icevision[all]
```
</div>

## Optional dependencies

### MMDetection Installation

We need to provide the appropriate version of the `mmcv-full` package as well as the `cuda` and the `torch` versions. Here are some examples for both the **CUDA** and the **CPU** versions  

!!! danger "Torch and CUDA version"  
    For the torch version use `torch.__version__` and replace the last number with 0.
    For the cuda version use: `torch.version.cuda`.

    Example: `TORCH_VERSION = torch1.8.0`; `CUDA_VERSION = cu101`

#### CUDA-Version Installation Example
<div class="termy">
```console
$ pip install mmcv-full=="1.3.3" -f https://download.openmmlab.com/mmcv/dist/CUDA_VERSION/TORCH_VERSION/index.html --upgrade
$ pip install mmdet
```
</div>

#### CPU-Version Installation
<div class="termy">
```console
$ pip install mmcv-full=="1.3.3+torch.1.8.0+cpu" -f https://download.openmmlab.com/mmcv/dist/index.html --upgrade
$ pip install mmdet
```
</div>


## Troubleshooting

### MMCV is not installing with cuda support
If you are installing MMCV from the wheel like described above and still are having problems with CUDA you will probably have to compile it locally. Do that by running:
```
pip install mmcv-full
```

If you encounter the following error it means you will have to install CUDA manually (the one that comes with conda installation will not do).
```
OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
```
Try installing it with:
```
sudo apt install nvidia-cuda-toolkit
```
Check the installation by running:
```
nvcc --version
```

### Error: Failed building wheel for pycocotools
If you encounter the following error, when installation process is building wheel for pycocotools:
```
unable to execute 'gcc': No such file or directory
error: command 'gcc' failed with exit status 1
```
Try installing gcc with:
```
sudo apt install gcc
```
Check the installation by running:
```
gcc --version
```
It should return something similar:
```
gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```
After that try installing icevision again.
