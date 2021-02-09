import pytest
from icevision.all import *


@pytest.mark.parametrize(
    "ds, model_type, path, config, weights_path",
    [
        (
            "fridge_ds",
            models.mmdet.faster_rcnn,
            "samples_source",
            "mmdet/faster_rcnn_r50_fpn_1x_coco.py",
            None,
        ),
        (
            "fridge_ds",
            models.mmdet.fcos,
            "samples_source",
            "mmdet/fcos_r50_caffe_fpn_gn-head_1x_coco.py",
            None,
        ),
        (
            "fridge_ds",
            models.mmdet.retinanet,
            "samples_source",
            "mmdet/retinanet_r50_fpn_1x_coco.py",
            None,
        ),
    ],
)
class TestBboxModels:  
    def test_mmdet_bbox_models_fastai(self, ds, model_type, path, config, weights_path, request):
        train_ds, valid_ds = request.getfixturevalue(ds)
        train_dl = model_type.train_dl(train_ds, batch_size=2)
        valid_dl = model_type.valid_dl(valid_ds, batch_size=2)
        
        config_path = request.getfixturevalue(path) / config
        
        model = model_type.model(config_path, num_classes=5, weights_path=weights_path)

        learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, splitter=fastai.trainable_params)
        learn.fine_tune(1, 3e-4)


    def test_mmdet_bbox_models_light(self, ds, model_type, path, config, weights_path, request):
        train_ds, valid_ds = request.getfixturevalue(ds)
        train_dl = model_type.train_dl(train_ds, batch_size=2)
        valid_dl = model_type.valid_dl(valid_ds, batch_size=2)
        
        config_path = request.getfixturevalue(path) / config
        
        model = model_type.model(config_path, num_classes=5, weights_path=weights_path)

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
