# Using different Faster RCNN backbones

In this example, we are training the Raccoon dataset using either [Fastai](https://github.com/fastai/fastai) or [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) training loop

=== "Fastai"
    ```python  hl_lines="70-76"

    # Installing IceVision
    # !pip install icevision[all]

    # Clone the raccoom dataset repository
    # !git clone https://github.com/datitran/raccoon_dataset

    # Imports
    from icevision.all import *

    # WARNING: Make sure you have already cloned the raccoon dataset using the command shown here above
    # Set images and annotations directories
    data_dir = Path("raccoon_dataset")
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"

    # Define class_map
    class_map = ClassMap(["raccoon"])

    # Parser: Use icevision predefined VOC parser
    parser = parsers.voc(
        annotations_dir=annotations_dir, images_dir=images_dir, class_map=class_map
    )

    # train and validation records
    train_records, valid_records = parser.parse()

    # Datasets
    # Transforms
    presize = 512
    size = 384
    train_tfms = tfms.A.Adapter(
        [*tfms.A.aug_tfms(size=size, presize=presize), tfms.A.Normalize()]
    )
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])

    # Train and Validation Dataset Objects
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    show_records(train_records[:3], ncols=3, class_map=class_map)

    # DataLoaders
    train_dl = faster_rcnn.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
    valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

    # Backbones
    backbone = backbones.resnet_fpn.resnet18(pretrained=True)
    # backbone = backbones.resnet_fpn.resnet34(pretrained=True)
    # backbone = backbones.resnet_fpn.resnet50(pretrained=True) # Default
    # backbone = backbones.resnet_fpn.resnet101(pretrained=True)
    # backbone = backbones.resnet_fpn.resnet152(pretrained=True)
    # backbone = backbones.resnet_fpn.resnext50_32x4d(pretrained=True)
    # backbone = backbones.resnet_fpn.resnext101_32x8d(pretrained=True)
    # backbone = backbones.resnet_fpn.wide_resnet50_2(pretrained=True)
    # backbone = backbones.resnet_fpn.wide_resnet101_2(pretrained=True)


    # Model
    model = faster_rcnn.model(backbone=backbone, num_classes=len(class_map))

    # Define metrics
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

    # fastai Learner
    learn = faster_rcnn.fastai.learner(
        dls=[train_dl, valid_dl], model=model, metrics=metrics
    )

    # Fastai Training
    # Learning Rate Finder
    learn.freeze()
    learn.lr_find()

    # Train using fastai fine tuning
    learn.fine_tune(20, lr=1e-4)

    # Inference
    infer_dl = faster_rcnn.infer_dl(valid_ds, batch_size=16)
    # Predict
    samples, preds = faster_rcnn.predict_dl(model, infer_dl)

    # Show some samples
    imgs = [sample["img"] for sample in samples]
    show_preds(
        imgs=imgs[:6],
        preds=preds[:6],
        class_map=class_map,
        denormalize_fn=denormalize_imagenet,
        ncols=3,
    )
    ```

=== "Pytorch Lightning"
    ```python  hl_lines="65-73"

    # Installing IceVision
    # !pip install icevision[all]

    # Clone the raccoom dataset repository
    # !git clone https://github.com/datitran/raccoon_dataset

    # Imports
    from icevision.all import *

    # WARNING: Make sure you have already cloned the raccoon dataset using the command shown here above
    # Set images and annotations directories
    data_dir = Path("raccoon_dataset")
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"

    # Define class_map
    class_map = ClassMap(["raccoon"])

    # Parser: Use icevision predefined VOC parser
    parser = parsers.voc(
        annotations_dir=annotations_dir, images_dir=images_dir, class_map=class_map
    )

    # train and validation records
    train_records, valid_records = parser.parse()

    # Datasets
    # Transforms
    presize = 512
    size = 384
    train_tfms = tfms.A.Adapter(
        [*tfms.A.aug_tfms(size=size, presize=presize), tfms.A.Normalize()]
    )
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])

    # Train and Validation Dataset Objects
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    show_records(train_records[:3], ncols=3, class_map=class_map)

    # DataLoaders
    train_dl = faster_rcnn.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
    valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

    # Backbones
    backbone = backbones.resnet_fpn.resnet18(pretrained=True)
    # backbone = backbones.resnet_fpn.resnet34(pretrained=True)
    # backbone = backbones.resnet_fpn.resnet50(pretrained=True) # Default
    # backbone = backbones.resnet_fpn.resnet101(pretrained=True)
    # backbone = backbones.resnet_fpn.resnet152(pretrained=True)
    # backbone = backbones.resnet_fpn.resnext50_32x4d(pretrained=True)
    # backbone = backbones.resnet_fpn.resnext101_32x8d(pretrained=True)
    # backbone = backbones.resnet_fpn.wide_resnet50_2(pretrained=True)
    # backbone = backbones.resnet_fpn.wide_resnet101_2(pretrained=True)


    # Model
    model = faster_rcnn.model(backbone=backbone, num_classes=len(class_map))

    # Define metrics
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

    # Train using pytorch-lightning
    class LightModel(faster_rcnn.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

    light_model = LightModel(model, metrics=metrics)

    trainer = pl.Trainer(max_epochs=20, gpus=1)
    trainer.fit(light_model, train_dl, valid_dl)

    # Inference
    infer_dl = faster_rcnn.infer_dl(valid_ds, batch_size=16)
    # Predict
    samples, preds = faster_rcnn.predict_dl(model, infer_dl)

    # Show some samples
    imgs = [sample["img"] for sample in samples]
    show_preds(
        imgs=imgs[:6],
        preds=preds[:6],
        class_map=class_map,
        denormalize_fn=denormalize_imagenet,
        ncols=3,
    )
    ```