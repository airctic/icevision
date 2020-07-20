__all__ = ["MantisEfficientDet"]


from mantisshrimp.imports import *
from mantisshrimp.core import *

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


# TODO: Can we map to all pretrained efficiendet models? (Not backbone in imagenet)
# checkpoint = torch.load("/home/lgvaz/Desktop/efficientdet_d0-f3276ba8.pth")
# net.load_state_dict(checkpoint)
def model(model_name: str, num_classes: int, img_size: int, pretrained: bool = True):
    """ Creates the model specific by model_name

    Args:
        model_name (str): Specifies the model to create, available options are: TODO
        num_classes (int): Number of classes of your dataset (including background)
        pretrained (int): If True, use a pretrained backbone (on ImageNet)

    Returns:
          nn.Module: The requested model
    """
    config = get_efficientdet_config(model_name=model_name)
    # TODO: Verify number of classes, last model layer seems to be outputing 36
    # units no matter what
    config.num_classes = num_classes  # Should we subtract one?
    config.image_size = img_size

    net = EfficientDet(config, pretrained_backbone=pretrained)
    net.class_net = HeadNet(
        config, num_outputs=num_classes, norm_kwargs=dict(eps=0.001, momentum=0.01),
    )

    return DetBenchTrain(net, config)


class MantisEfficientDet(nn.Module):
    def __init__(self, num_classes: int, img_size: int, model_name="efficientdet_d0"):
        super().__init__()
        self.model = model(
            num_classes=num_classes, img_size=img_size, model_name=model_name
        )

    def forward(self, inputs, targets):
        return self.model(inputs, targets)

    @staticmethod
    def build_train_sample(
        img: np.ndarray, labels: List[int], bboxes: List[BBox], **kwargs
    ):
        x = im2tensor(img)

        y = {
            "cls": tensor(labels, dtype=torch.int64),
            "boxes": tensor([bbox.yxyx for bbox in bboxes], dtype=torch.float64),
        }

        return x, y

    @staticmethod
    def build_valid_sample(
        img: np.ndarray, labels: List[int], bboxes: List[BBox], **kwargs
    ):
        x, y = MantisEfficientDet.build_train_sample(
            img=img, labels=labels, bboxes=bboxes, **kwargs
        )

        return x, y

    @classmethod
    def collate_fn(cls, samples):
        train_samples = [cls.build_train_sample(**sample) for sample in samples]
        xb, yb = zip(*train_samples)
        # TODO HACK
        yb2 = {}
        yb2["bbox"] = [o["boxes"].float() for o in yb]
        yb2["cls"] = [o["cls"].float() for o in yb]
        return torch.stack(xb), yb2

    @classmethod
    def valid_collate_fn(cls, samples):
        valid_samples = [cls.build_valid_sample(**sample) for sample in samples]
        xb, yb = zip(*valid_samples)
        # TODO HACK
        yb2 = {}
        yb2["bbox"] = [o["boxes"].float() for o in yb]
        yb2["cls"] = [o["cls"].float() for o in yb]

        batch_size = len(valid_samples)
        yb2["img_scale"] = tensor([1.0] * batch_size, dtype=torch.float)
        yb2["img_size"] = tensor([xb[0].shape[-2:]] * batch_size, dtype=torch.float)

        return torch.stack(xb), yb2

    @classmethod
    def dataloader(cls, dataset, **kwargs):
        return DataLoader(dataset=dataset, collate_fn=cls.collate_fn, **kwargs)

    @classmethod
    def valid_dataloader(cls, dataset, **kwargs):
        return DataLoader(dataset=dataset, collate_fn=cls.valid_collate_fn, **kwargs)

    @staticmethod
    def loss(preds, targets) -> Tensor:
        return preds["loss"]
