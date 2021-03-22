pip install git+git://github.com/airctic/icevision.git#egg=icevision[all] --upgrade
pip install git+git://github.com/airctic/icedata.git --upgrade

pip install mmcv-full=="1.2.5" -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html --upgrade
#pip install mmdet --upgrade

# Install mmdetection via cloning because of config folder
git clone https://github.com/open-mmlab/mmdetection.git
pip install -e mmdetection -q
