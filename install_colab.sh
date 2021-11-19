echo "- Installing mmcv"
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html --upgrade -q  

echo "- Installing mmdet"
pip install mmdet==2.17.0 --upgrade -q


echo "- Installing icevision from master"
pip install git+git://github.com/airctic/icevision.git#egg=icevision[all] -U -q
echo "- Installing icedata from master"      
pip install git+git://github.com/airctic/icedata.git -U -q
echo "- Installing yolov5-icevision" 
pip install git+git://github.com/airctic/yolov5-icevision.git -U -q 


# Installation completed
echo "Installation completed!"


# Restart the Kernel
echo "=========================="
echo "Please Restart Kernel!"
echo "=========================="