### 0.8

- Yolov5 support
- fastai new version (multi-gpu and half-precision)
- mmdet new version
- support for negative samples
- pytorch 1.8

### namespace

- old -
model_type = models.mmdet.retinanet
backbone = model_type.backbones.resnet50_fpn_1x

models.torchvision.backbones

model_type = models.torchvision.faster_rcnn
backbone = model_type.backbones.resnet50



- current -
import icevision as ice

model_type = ice.detection.models.mmdet.retinanet
backbone = model_type.backbones.timm.resnet18

model_type = ice.classification.models.mmdet.?
backbone = model_type.backbones.timm.resnet18? (no backbone for classification?)


model_type = ice.multitask


### Mmdet backbone

- two steps
1. register the backbones
 - Files to register the backbones

2. specify backbones per model


base_config_path = mmdet_configs_path / "retinanet"
config_path=base_config_path / "retinanet_r50_fpn_1x_coco.py"
config_path
cfg = Config.fromfile(config_path)
cfg.model.backbone = {
    'type': 'MobileNetv3Large100', 
}

cfg.model.neck.in_channels = neck_in_channels  # [16, 24, 40 , 112, 960]

cfg.model.bbox_head.num_classes = len(parser.class_map) - 1

model = build_detector(cfg.model, cfg.get("train_cfg"), cfg.get("test_cfg"))
model

def get_in_channels():
  x = torch.randn(1,3,224,224)
  features = backbone(x)
  neck_in_channels = [f.shape[1] for f in features]
  return neck_in_channels