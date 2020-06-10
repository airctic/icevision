from mantisshrimp.backbones import *

def test_torchvision_backbones():
    supported_backbones = ["mobile_net", "vgg_11", "vgg_13", "vgg_16", "vgg_19", 
    "resnet_18", "resnet_34", "resnet_50", "resnet_101", "resnet_152", "resnext101_32x8d"]
    pretrained_status = [True, False]

    for backbone in supported_backbones:
        for is_pretrained in pretrained_status:
            model = create_torchvision_backbone(backbone=backbone, pretrained=is_pretrained)
            assert isinstance(model, torch.nn.modules.container.Sequential)



