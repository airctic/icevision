echo "Installing icevision + dependencices for CUDA 10"
echo "- Installing torch and its dependencies"
pip install torchtext==0.9.0 --upgrade -q
pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html --upgrade -q
echo "- Installing fastai"
pip install fastai==2.3.1 --upgrade -q
echo "- Installing icevision from master"
pip install git+git://github.com/airctic/icevision.git\#egg=icevision[all] --upgrade -q
echo "- Installing icedata from master"      
pip install git+git://github.com/airctic/icedata.git --upgrade -q
echo "- Installing yolov5-icevision" 
pip install yolov5-icevision --upgrade -q 
echo "- Installing mmcv"
pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html --upgrade -q
echo "- Installing mmdet"
pip install mmdet==2.13.0 --upgrade -q
echo "icevision installation finished!"  

# restart notebook
echo "Restarting runtime!"
kill -9 -1