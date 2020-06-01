__all__ = ["MantisDetr"]

from ..imports import *
from ..utils import *
from ..core import *
from .mantis_module import *
from .detr_demo import *


class MantisDetr(MantisModule):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = DETRdemo(num_classes=91)
        if pretrained:
            self._load_pretrained_weights()

    def forward(self, x):
        return self.model(x)

    def predict(self, img, thresh=0.7):
        assert len(img.shape) == 3, "Only batch size 1 supported"
        img_h, img_w = img.shape[1:]
        # get predictions
        outs = self(img[None])
        probs = outs["pred_logits"].softmax(-1)[0, :, :-1]
        bboxes = outs["pred_boxes"][0]
        # calculate what predictions to keep
        max_probs = probs.max(-1)[0]
        keep = max_probs > thresh
        # filter predictions to keep
        keep_probs = to_np(probs[keep])
        keep_bboxes = [
            BBox.from_relative_xcycwh(*points, img_w, img_h)
            for points in to_np(bboxes[keep])
        ]
        return keep_probs, keep_bboxes

    def _load_pretrained_weights(self):
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth",
            check_hash=True,
        )
        self.model.load_state_dict(state_dict)

    def dataloader(cls, **kwargs) -> DataLoader:
        pass
