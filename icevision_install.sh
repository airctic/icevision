# Pick your installation target: cuda11 or cuda10 or cpu
# Pick icevision version: if empty, the PyPi release version will be chosen. If you pass `master`, the GitHub master version will be chosen

# Examples
## Install cuda11  and icevsision master version
# !bash icevision_install.sh cuda11 master  

## Install cpu and icevsision PyPi version
# !bash icevision_install.sh cpu 

target="${1}" 
case ${target} in
   cuda11)  
      echo "Installing icevision + dependencices for ${1}"
      echo "- Installing torch and its dependencies"
      pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 --upgrade

      echo "- Installing mmcv"
      pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
    ;;
    
   cpu)  
      echo "Installing icevision + dependencices for ${1}"
      echo "- Installing torch and its dependencies"
      pip install torch==1.13.1+cpu torchvision==0.14.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu --upgrade

      echo "- Installing mmcv"
      pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13/index.html --upgrade -q
    ;;

   *)  
      echo "Coud not install icevision. Check out which torch and torchvision versions are compatible with your CUDA version" 
      exit -1 # Command to come out of the program with status -1
      ;; 
esac


echo "- Installing mmdet"
pip install mmdet==2.26.0 --upgrade -q

echo "- Installing mmseg"
pip install mmsegmentation==0.29.1 --upgrade -q

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