!!! danger "Important"  
    We currently only support Linux/MacOS installations


## 1. Installing torch
Depending on what version of cuda driver you'd like to use, you can install different versions of torch builds. If you're not sure which version to choose, we advise to use the current torch default `cuda-10.2`

=== "cuda-10.2"
    ```
    pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
    ```

=== "cuda-11.3"
    ```
    pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    ```
=== "cpu"
    ```
    pip install torch==1.10.0+cpu torchvision==0.11.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
    ```

??? note "checking your `torch`-`cuda` version"  
    To see what version of `torch` and `cuda` is installed in your current environment, run:
    ```
    python -c "import torch;print(torch.__version__)"
    ```
    output:
    ```
    1.10.1+cu102
    ```
    Your installed torch version will determine which version of `mmcv-full` you can install.

## 2. Installing mmcv-full (*optional*)

Installing `mmcv-full` is optional, yet it will let you unleash the full potential of `icevision` and allow you to use the large library of models available in `mmdet`, therefore we strongly recommend doing it.

=== "cuda-10.2"
    ```
    pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
    pip install mmdet==2.17.0
    ```

=== "cuda-11.3"
    ```
    pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
    pip install mmdet==2.17.0
    ```

=== "cpu"
    ```
    pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html
    pip install mmdet==2.17.0
    ```


??? "testing `mmcv` installation"
    
    Installing `mmcv-full` can be tricky as it depends on both the exact `torch` and `cuda` version. 
    We highly recommend that you test your installation. You can verify it by executing the following command inside your virtual environment:
    ```bash
    curl -sSL https://raw.githubusercontent.com/open-mmlab/mmcv/master/.dev_scripts/check_installation.py | python -
    ```
    
    &nbsp;
    If everything went fine, you should see something like the following:
    
    ```
    Start checking the installation of mmcv-full ...
    CPU ops were compiled successfully.
    CUDA ops were compiled successfully.
    mmcv-full has been installed successfully.
    
    Environment information:
    -----------------------------------------------------------
    sys.platform: linux
    Python: 3.8.12 (default, Oct 12 2021, 13:49:34) [GCC 7.5.0]
    CUDA available: True
    GPU 0: GeForce RTX 2060
    CUDA_HOME: /usr/local/cuda
    NVCC: Build cuda_11.1.TC455_06.29069683_0
    GCC: gcc (Ubuntu 9.3.0-17ubuntu1~20.04) 9.3.0
    PyTorch: 1.10.0+cu111
    PyTorch compiling details: PyTorch built with:
        - GCC 7.3
        - C++ Version: 201402
        - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
        - Intel(R) MKL-DNN v2.2.3 (Git Hash 7336ca9f055cf1bfa13efb658fe15dc9b41f0740)
        - OpenMP 201511 (a.k.a. OpenMP 4.5)
        - LAPACK is enabled (usually provided by MKL)
        - NNPACK is enabled
        - CPU capability usage: AVX2
        - CUDA Runtime 11.1
        - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
        - CuDNN 8.0.5
        - Magma 2.5.2
        - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.1, CUDNN_VERSION=8.0.5, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.10.0, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, 
    
    TorchVision: 0.11.1+cu111
    OpenCV: 4.5.4
    MMCV: 1.3.17
    MMCV Compiler: GCC 7.3
    MMCV CUDA Compiler: 11.1
    -----------------------------------------------------------
    ```
    
    
    &nbsp;

## 3. Installing icevision
- inference


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
