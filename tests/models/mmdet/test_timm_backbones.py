import pytest
from icevision.all import *
from icevision.models.mmdet.utils import *


# @pytest.mark.parametrize(
#     "ds, model_type",
#     [
#         (
#             "fridge_ds",
#             models.mmdet.retinanet,
#         ),
#     ],
# )
@pytest.mark.parametrize(
    "ds, model_type, backbone",
    (
        (
            "fridge_ds",
            models.mmdet.retinanet,
            models.mmdet.retinanet.backbones.timm.mobilenet.mobilenetv3_large_100,
        ),
        (
            "fridge_ds",
            models.mmdet.retinanet,
            models.mmdet.retinanet.backbones.timm.resnet.resnet50,
        ),
    ),
)
class TestTimmBackbones:
    def dls_model(self, ds, model_type, backbone, samples_source, request):
        train_ds, valid_ds = request.getfixturevalue(ds)
        train_dl = model_type.train_dl(train_ds, batch_size=2)
        valid_dl = model_type.valid_dl(valid_ds, batch_size=2)

        # backbone = model_type.backbone
        backbone.config_path = samples_source / backbone.config_path

        model = model_type.model(backbone=backbone(pretrained=True), num_classes=5)

        return train_dl, valid_dl, model

    def test_mmdet_bbox_models_fastai(
        self, ds, model_type, backbone, samples_source, request
    ):
        train_dl, valid_dl, model = self.dls_model(
            ds, model_type, backbone, samples_source, request
        )

        learn = model_type.fastai.learner(
            dls=[train_dl, valid_dl], model=model, splitter=fastai.trainable_params
        )
        learn.fine_tune(1, 3e-4)

    def test_mmdet_bbox_models_light(
        self, ds, model_type, backbone, samples_source, request
    ):
        train_dl, valid_dl, model = self.dls_model(
            ds, model_type, backbone, samples_source, request
        )

        class LitModel(model_type.lightning.ModelAdapter):
            def configure_optimizers(self):
                return Adam(self.parameters(), lr=1e-4)

        light_model = LitModel(model)
        trainer = pl.Trainer(
            max_epochs=1,
            weights_summary=None,
            num_sanity_val_steps=0,
            logger=False,
            checkpoint_callback=False,
        )
        trainer.fit(light_model, train_dl, valid_dl)


@pytest.mark.parametrize(
    "ds, model_type, backbone",
    (
        (
            "fridge_ds",
            models.mmdet.fcos,
            MMDetTimmBackboneConfig(
                model_name="fcos",
                config_path="fcos_r50_caffe_fpn_gn-head_1x_coco.py",
                backbone_dict={
                    "type": "ResNet50_Timm",
                },
            ),
        ),
        (
            "fridge_ds",
            models.mmdet.fcos,
            MMDetTimmBackboneConfig(
                model_name="fcos",
                config_path="fcos_r50_caffe_fpn_gn-head_1x_coco.py",
                backbone_dict={
                    "type": "ResNet50_Timm",
                    "pretrained": True,
                    "out_indices": (2, 3, 4),
                    "norm_eval": True,
                    "frozen_stem": True,
                    "frozen_stages": 1,
                },
                weights_url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet50_ram-a26f946b.pth",
            ),
        ),
    ),
)
class TestTimmBackbonesWithConfig:
    def dls_model(self, ds, model_type, backbone, samples_source, request):
        train_ds, valid_ds = request.getfixturevalue(ds)
        train_dl = model_type.train_dl(train_ds, batch_size=2)
        valid_dl = model_type.valid_dl(valid_ds, batch_size=2)

        backbone.config_path = samples_source / backbone.config_path

        model = model_type.model(backbone=backbone(pretrained=True), num_classes=5)

        return train_dl, valid_dl, model

    def test_mmdet_bbox_models_fastai(
        self, ds, model_type, backbone, samples_source, request
    ):
        train_dl, valid_dl, model = self.dls_model(
            ds, model_type, backbone, samples_source, request
        )

        learn = model_type.fastai.learner(
            dls=[train_dl, valid_dl], model=model, splitter=fastai.trainable_params
        )
        learn.fine_tune(1, 3e-4)

    def test_mmdet_bbox_models_light(
        self, ds, model_type, backbone, samples_source, request
    ):
        train_dl, valid_dl, model = self.dls_model(
            ds, model_type, backbone, samples_source, request
        )

        class LitModel(model_type.lightning.ModelAdapter):
            def configure_optimizers(self):
                return Adam(self.parameters(), lr=1e-4)

        light_model = LitModel(model)
        trainer = pl.Trainer(
            max_epochs=1,
            weights_summary=None,
            num_sanity_val_steps=0,
            logger=False,
            checkpoint_callback=False,
        )
        trainer.fit(light_model, train_dl, valid_dl)
