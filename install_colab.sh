pip install openmim -q
echo "- Installing mmcv"
mim install mmcv-full==1.3.13
echo "- Installing mmdet"
mim install mmdet==2.12.0
echo "- Installing mmseg"
mim install mmsegmentation==0.17.0

echo "- Installing icevision from master"
pip install git+git://github.com/airctic/icevision.git#egg=icevision[all] -U -q
echo "- Installing icedata from master"      
pip install git+git://github.com/airctic/icedata.git -U -q
echo "- Installing yolov5-icevision" 
pip install yolov5-icevision -U -q 


# Installation completed
echo "Installation completed!"


# Restart the Kernel
echo "=========================="
echo "Please Restart Kernel!"
echo "=========================="