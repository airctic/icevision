__all__ = ['MantisMaskRCNN']

from ..imports import *
from .mantis_rcnn import *
from .rcnn_param_groups import *

class MantisMaskRCNN(MantisRCNN):
    @delegates(MaskRCNN.__init__)
    def __init__(self, n_class, h=256, pretrained=True, metrics=None, **kwargs):
        store_attr(self, 'n_class,h,pretrained,kwargs')
        super().__init__(metrics=metrics)

    def forward(self, images, targets=None): return self.m(images, targets)

    def _create_model(self):
        self.m = maskrcnn_resnet50_fpn(pretrained=self.pretrained, **self.kwargs)
        in_features = self.m.roi_heads.box_predictor.cls_score.in_features
        self.m.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.n_class)
        in_features_mask = self.m.roi_heads.mask_predictor.conv5_mask.in_channels
        self.m.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, self.h, self.n_class)

    def _get_pgs(self): return rcnn_param_groups(self.m)