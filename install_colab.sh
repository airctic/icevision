echo "Installing icevision + dependencices for CUDA 10"
echo "Uninstalling some dependencies to prevent errors"
pip uninstall torchvision -y
pip uninstall fastai -y

 echo "Installing some dependencies to prevent errors"
pip install PyYAML>=5.1 -U -q
pip install datascience -U -q
pip install tensorflow==2.4.0 -U -q
pip install google-colab -U -q

echo "- Installing torch and its dependencies"
echo "- Installing torch and its dependencies"
pip install torchtext==0.9.0 -U -q
pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -U -q

echo "- Installing fastai"
pip install fastai==2.3.1 -U -q

echo "- Installing icevision from master"
pip install git+git://github.com/airctic/icevision.git@fix-colab-install -U -q
echo "- Installing icedata from master"      
pip install git+git://github.com/airctic/icedata.git -U -q
echo "- Installing yolov5-icevision" 
pip install yolov5-icevision -U -q 
echo "- Installing mmcv"
pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html -U -q
echo "- Installing mmdet"
pip install mmdet==2.13.0 -U -q
echo "icevision installation finished"  

# restart notebook
echo "Restarting runtime!"
kill -9 -1