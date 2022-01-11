platform="${1}" 
case ${platform} in 
   colab)
      echo "- Installing icevision inference in Colab"  
      echo "Uninstalling some dependencies to prevent errors"
      pip uninstall fastai -y
      pip uninstall torchvision -y
      pip uninstall torchtext -y

      echo "- Updating torchvision"
      pip install torchvision==0.9.0 -U -q
      
      echo "- Updating torchtext"
      pip install torchtext==0.9.0 -U -q 

      echo "Installing some dependencies to prevent errors"
      pip install PyYAML>=5.1 -U -q
      pip install datascience -U -q
      pip install tensorflow==2.4.0 -U -q
      pip install google-colab -U -q
      
      
      echo "- Installing icevision and icedata"
      pip install git+https://github.com/airctic/icevision.git#egg=icevision[inference] -U -q
      pip install icedata -U -q
      echo "icevision installation finished!"   
      ;; 
   *)  # Installing only icevision and icedata
      echo "- Installing icevision inference"
      pip install git+https://github.com/airctic/icevision.git#egg=icevision[inference] -U -q
      pip install icedata -U -q
      echo "icevision installation finished!"
      ;; 
esac
# a workaround regarding opencv in colab issue: https://github.com/airctic/icevision/issues/1012
pip install opencv-python-headless==4.1.2.30