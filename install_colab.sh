!pip install openmim -q
!echo "- Installing mmcv"
!mim install mmcv-full
!echo "- Installing mmdet"
!mim install mmdet


!echo "- Installing icevision from master"
!pip install git+git://github.com/airctic/icevision.git#egg=icevision[all] -U -q
!echo "- Installing icedata from master"      
!pip install git+git://github.com/airctic/icedata.git -U -q
!echo "- Installing yolov5-icevision" 
!pip install yolov5-icevision -U -q 


# restart notebook
!echo "Restarting runtime!"
exit()
