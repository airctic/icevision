# Pick your installation target: cuda11 or cuda10 or cpu
# Pick icevision version: if empty, the PyPi release version will be chosen. If you pass `master`, the GitHub master version will be chosen

# Examples
## Install cuda11  and icevsision master version
# !bash icevision_install.sh cuda11 master  

## Install cpu and icevsision PyPi version
# !bash icevision_install.sh cpu 

target="${1}" 
case ${target} in 
   cuda10)  
      echo "Installing icevision + dependencices for ${1}"
      echo "- Installing torch and its dependencies"
      pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 torchtext==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html --upgrade -q

      echo "- Installing mmcv"
      pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html --upgrade -q    
      ;; 

   cuda11)  
      echo "Installing icevision + dependencices for ${1}"
      echo "- Installing torch and its dependencies"
      pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchtext==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html --upgrade 

      echo "- Installing mmcv"
      pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html --upgrade -q  
    ;;  
    
   cpu)  
      echo "Installing icevision + dependencices for ${1}"
      echo "- Installing torch and its dependencies"
      pip install torch=="1.10.0+cpu" torchvision=="0.11.1+cpu" torchtext==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html  

      echo "- Installing mmcv"
      pip install mmcv-full=="1.3.17" -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html --upgrade -q  
    ;;

   *)  
      echo "Coud not install icevision. Check out which torch and torchvision versions are compatible with your CUDA version" 
      exit -1 # Command to come out of the program with status -1
      ;; 
esac


echo "- Installing mmdet"
pip install mmdet==2.17.0 --upgrade -q

echo "- Installing mmseg"
pip install mmsegmentation==0.20.2 --upgrade -q

icevision_version="${2}"

case ${icevision_version} in 
   master)
      echo "- Installing icevision from master"
      pip install git+https://github.com/airctic/icevision.git#egg=icevision[all] --upgrade -q

      echo "- Installing icedata from master"      
      pip install git+https://github.com/airctic/icedata.git --upgrade -q
      ;;

   *) 
      echo "- Installing icevision from PyPi"
      pip install icevision[all] --upgrade -q

      echo "- Installing icedata from PyPi"      
      pip install icedata --upgrade -q
      ;;
  esac

# a workaround regarding opencv in colab issue: https://github.com/airctic/icevision/issues/1012
pip install opencv-python-headless==4.1.2.30
echo "icevision installation finished!"  