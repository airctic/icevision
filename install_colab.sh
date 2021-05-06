pip install git+git://github.com/airctic/icevision.git\#egg=icevision[all] --upgrade
pip install git+git://github.com/airctic/icedata.git --upgrade

pip install mmcv-full=="1.3.3" -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html --upgrade
pip install mmdet --upgrade

pip install yolov5-icevision --upgrade

# restart notebook
kill -9 -1
