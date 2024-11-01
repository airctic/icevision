import pytest
import random
from icevision.all import *


@pytest.mark.parametrize(
    "ds, model_type, backbone_name, pretrained, cfg_options",
    (
        ("fridge_ds", models.mmdet.vfnet, "resnet50_fpn_mstrain_2x", False, None),
        # ("fridge_ds", models.mmdet.retinanet, "resnet50_fpn_1x", True, None),
        # ("fridge_ds", models.mmdet.retinanet, "resnet50_fpn_1x", False, None),
        # ("fridge_ds", models.mmdet.faster_rcnn, "resnet101_fpn_2x", False, None),
        ("fridge_ds", models.mmdet.ssd, "ssd300", False, None),
        # ("fridge_ds", models.mmdet.yolox, "yolox_s_8x8", False, None),
        # ("fridge_ds", models.mmdet.yolof, "yolof_r50_c5_8x8_1x_coco", False, None),
        # ("fridge_ds", models.mmdet.detr, "r50_8x2_150e_coco", False, None),
        # ("fridge_ds", models.mmdet.deformable_detr, "twostage_refine_r50_16x2_50e_coco", False, None),
        # ("fridge_ds", models.mmdet.fsaf, "x101_64x4d_fpn_1x_coco", False, None),
        # ("fridge_ds", models.mmdet.sabl, "r101_fpn_gn_2x_ms_640_800_coco", False, None),
        # ("fridge_ds", models.mmdet.centripetalnet, "hourglass104_mstest_16x6_210e_coco", False, None),
        # (
        #     "fridge_ds",
        #     models.mmdet.retinanet,
        #     False,
        #     {
        #         "model.bbox_head.loss_bbox.loss_weight": 2,
        #         "model.bbox_head.loss_cls.loss_weight": 0.8,
        #     },
        # ),
    ),
)
class TestBboxModels:
    def dls_model(
        self, ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
    ):
        train_ds, valid_ds = request.getfixturevalue(ds)
        train_dl = model_type.train_dl(train_ds, batch_size=2)
        valid_dl = model_type.valid_dl(valid_ds, batch_size=2)

        backbone = getattr(model_type.backbones, backbone_name)()
        backbone.config_path = samples_source / backbone.config_path

        model = model_type.model(
            backbone=backbone(pretrained=pretrained),
            num_classes=5,
            cfg_options=cfg_options,
        )

        return train_dl, valid_dl, model

    @pytest.mark.skip
    def test_mmdet_bbox_models_fastai(
        self, ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
    ):
        train_dl, valid_dl, model = self.dls_model(
            ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
        )

        learn = model_type.fastai.learner(
            dls=[train_dl, valid_dl], model=model, splitter=fastai.trainable_params
        )
        learn.fine_tune(1, 3e-4)

    def test_mmdet_bbox_models_light_train(
        self, ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
    ):
        train_dl, valid_dl, model = self.dls_model(
            ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
        )

        class LitModel(model_type.lightning.ModelAdapter):
            def configure_optimizers(self):
                return Adam(self.parameters(), lr=1e-4)

        light_model = LitModel(model)
        trainer = pl.Trainer(
            max_epochs=1,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(light_model, train_dl, valid_dl)


    @pytest.mark.skip
    def test_mmdet_bbox_models_light_training_step_returns_loss(
        self, ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
    ):
        with torch.set_grad_enabled(True):
            train_dl, _, model = self.dls_model(
                ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
            )

            class LitModel(model_type.lightning.ModelAdapter):
                def configure_optimizers(self):
                    return Adam(self.parameters(), lr=1e-4)

            expected_loss = random.randint(0, 10)
            model.train_step = lambda **args: {"log_vars": {}, "loss": expected_loss}
            light_model = LitModel(model)
            for batch in train_dl:
                batch
                break

            loss = light_model.training_step(batch, 0)

            assert loss == expected_loss

    @pytest.mark.skip
    def test_mmdet_bbox_models_light_logs_losses_during_training_step(
        self, ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
    ):
        with torch.set_grad_enabled(True):
            train_dl, _, model = self.dls_model(
                ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
            )

            class LitModel(model_type.lightning.ModelAdapter):
                def __init__(self, model, metrics=None):
                    super(LitModel, self).__init__(model, metrics)
                    self.model = model
                    self.logs = {}

                def configure_optimizers(self):
                    return Adam(self.parameters(), lr=1e-4)

                def log(self, key, value, **args):
                    super(LitModel, self).log(key, value, **args)
                    self.logs[key] = value

            expected_loss_key = "my_loss_key"
            model.train_step = lambda **args: {
                "log_vars": {expected_loss_key: random.randint(0, 10)},
                "loss": random.randint(0, 10),
            }
            light_model = LitModel(model)
            for batch in train_dl:
                batch
                break

            light_model.training_step(batch, 0)

            assert list(light_model.logs.keys()) == [f"train_{expected_loss_key}"]


    @pytest.mark.skip
    def test_mmdet_bbox_models_light_validate(
        self, ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
    ):
        _, valid_dl, model = self.dls_model(
            ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
        )

        class LitModel(model_type.lightning.ModelAdapter):
            def configure_optimizers(self):
                return Adam(self.parameters(), lr=1e-4)

        light_model = LitModel(model)
        trainer = pl.Trainer(
            max_epochs=1,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.validate(light_model, valid_dl)


    @pytest.mark.skip
    def test_mmdet_bbox_models_light_logs_losses_during_validation_step(
        self, ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
    ):
        with torch.set_grad_enabled(False):
            _, valid_dl, model = self.dls_model(
                ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
            )

            class LitModel(model_type.lightning.ModelAdapter):
                def __init__(self, model, metrics=None):
                    super(LitModel, self).__init__(model, metrics)
                    self.model = model
                    self.logs = {}

                def configure_optimizers(self):
                    return Adam(self.parameters(), lr=1e-4)

                def log(self, key, value, **args):
                    super(LitModel, self).log(key, value, **args)
                    self.logs[key] = value

            expected_loss_key = "my_loss_key"
            model.train_step = lambda **args: {
                "log_vars": {expected_loss_key: random.randint(0, 10)}
            }
            light_model = LitModel(model)
            for batch in valid_dl:
                batch
                break

            light_model.validation_step(batch, 0)

            assert list(light_model.logs.keys()) == [f"val_{expected_loss_key}"]

    @pytest.mark.skip
    def test_mmdet_bbox_models_light_finalizes_metrics_on_validation_epoch_end(
        self, ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
    ):
        with torch.set_grad_enabled(False):
            _, _, model = self.dls_model(
                ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
            )

            class LitModel(model_type.lightning.ModelAdapter):
                def __init__(self, model, metrics=None):
                    super(LitModel, self).__init__(model, metrics)
                    self.was_finalize_metrics_called = False

                def configure_optimizers(self):
                    return Adam(self.parameters(), lr=1e-4)

                def finalize_metrics(self):
                    self.was_finalize_metrics_called = True

            light_model = LitModel(model)

            light_model.validation_epoch_end(None)

            assert light_model.was_finalize_metrics_called == True

    @pytest.mark.skip
    def test_mmdet_bbox_models_light_test(
        self, ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
    ):
        _, valid_dl, model = self.dls_model(
            ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
        )

        class LitModel(model_type.lightning.ModelAdapter):
            def configure_optimizers(self):
                return Adam(self.parameters(), lr=1e-4)

        light_model = LitModel(model)
        trainer = pl.Trainer(
            max_epochs=1,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.test(light_model, valid_dl)

    @pytest.mark.skip
    def test_mmdet_bbox_models_light_logs_losses_during_test_step(
        self, ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
    ):
        with torch.set_grad_enabled(False):
            _, valid_dl, model = self.dls_model(
                ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
            )

            class LitModel(model_type.lightning.ModelAdapter):
                def __init__(self, model, metrics=None):
                    super(LitModel, self).__init__(model, metrics)
                    self.model = model
                    self.logs = {}

                def configure_optimizers(self):
                    return Adam(self.parameters(), lr=1e-4)

                def log(self, key, value, **args):
                    super(LitModel, self).log(key, value, **args)
                    self.logs[key] = value

            expected_loss_key = "my_loss_key"
            model.train_step = lambda **args: {
                "log_vars": {expected_loss_key: random.randint(0, 10)}
            }
            light_model = LitModel(model)
            for batch in valid_dl:
                batch
                break

            light_model.test_step(batch, 0)

            assert list(light_model.logs.keys()) == [f"test_{expected_loss_key}"]

    @pytest.mark.skip
    def test_mmdet_bbox_models_light_finalizes_metrics_on_test_epoch_end(
        self, ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
    ):
        with torch.set_grad_enabled(False):
            _, _, model = self.dls_model(
                ds, model_type, backbone_name, pretrained, cfg_options, samples_source, request
            )

            class LitModel(model_type.lightning.ModelAdapter):
                def __init__(self, model, metrics=None):
                    super(LitModel, self).__init__(model, metrics)
                    self.was_finalize_metrics_called = False

                def configure_optimizers(self):
                    return Adam(self.parameters(), lr=1e-4)

                def finalize_metrics(self):
                    self.was_finalize_metrics_called = True

            light_model = LitModel(model)

            light_model.test_epoch_end(None)

            assert light_model.was_finalize_metrics_called == True


@pytest.fixture()
def mask_dls_model(coco_mask_records, samples_source):
    train_records, valid_records = coco_mask_records[:2], coco_mask_records[:1]

    presize, size = 256, 128
    train_tfms = tfms.A.Adapter(
        [*tfms.A.aug_tfms(presize=presize, size=size), tfms.A.Normalize()]
    )
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])

    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    model_type = models.mmdet.mask_rcnn
    train_dl = model_type.train_dl(train_ds, batch_size=2)
    valid_dl = model_type.valid_dl(valid_ds, batch_size=2)

    backbone = model_type.backbones.resnet50_fpn_1x()
    backbone.config_path = samples_source / backbone.config_path

    model = model_type.model(backbone=backbone, num_classes=81)

    return train_dl, valid_dl, model, model_type


@pytest.mark.skip
def test_mmdet_mask_models_fastai(mask_dls_model):
    train_dl, valid_dl, model, model_type = mask_dls_model

    learn = model_type.fastai.learner(
        dls=[train_dl, valid_dl], model=model, splitter=fastai.trainable_params
    )
    learn.fine_tune(1, 3e-4)


@pytest.mark.skip
def test_mmdet_mask_models_light(mask_dls_model):
    train_dl, valid_dl, model, model_type = mask_dls_model

    class LitModel(model_type.lightning.ModelAdapter):
        def configure_optimizers(self):
            return Adam(self.parameters(), lr=1e-4)

    light_model = LitModel(model)
    trainer = pl.Trainer(
        max_epochs=1,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(light_model, train_dl, valid_dl)
