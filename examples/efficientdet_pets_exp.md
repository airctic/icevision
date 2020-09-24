# How to use EffecientDet

In this example, we show how to train an EffecientDet model on the PETS dataset using either [Fastai](https://github.com/fastai/fastai) or [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) training loop

=== "Fastai"
    ```python  hl_lines="52-61"

    # Installing IceVision
    # !pip install icevision[all] icedata

    # Imports
    from icevision.all import *
    import icedata

    # Common part to all models

    # Loading Data
    data_dir = icedata.pets.load_data()

    # Parser
    class_map = icedata.pets.class_map()
    parser = icedata.pets.parser(data_dir, class_map)
    train_records, valid_records = parser.parse()
    show_records(train_records[:3], ncols=3, class_map=class_map)

    # Datasets
    # Transforms
    presize = 512
    size = 384
    train_tfms = tfms.A.Adapter(
        [*tfms.A.aug_tfms(size=size, presize=presize), tfms.A.Normalize()]
    )
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])

    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    samples = [train_ds[0] for _ in range(3)]
    show_samples(samples, ncols=3, class_map=class_map, denormalize_fn=denormalize_imagenet)

    # EffecientDet Specific Part

    # DataLoaders
    train_dl = efficientdet.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
    valid_dl = efficientdet.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)
    batch, samples = first(train_dl)
    show_samples(
        samples[:6], class_map=class_map, ncols=3, denormalize_fn=denormalize_imagenet
    )

    # Model
    model = efficientdet.model(
        model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size
    )

    # Define metrics
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

    # Fastai Learner
    learn = efficientdet.fastai.learner(
        dls=[train_dl, valid_dl], model=model, metrics=metrics
    )

    # Fastai Training
    learn.freeze()
    learn.lr_find()

    learn.fine_tune(10, 1e-2, freeze_epochs=1)

    # Inference
    # DataLoader
    infer_dl = efficientdet.infer_dl(valid_ds, batch_size=8)
    # Predict
    samples, preds = efficientdet.predict_dl(model, infer_dl)
    # Show samples
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
    ```python  hl_lines="52-60"
    
    # Installing IceVision
    # !pip install icevision[all] icedata

    # Imports
    from icevision.all import *
    import icedata

    # Common part to all models

    # Loading Data
    data_dir = icedata.pets.load_data()

    # Parser
    class_map = icedata.pets.class_map()
    parser = icedata.pets.parser(data_dir, class_map)
    train_records, valid_records = parser.parse()
    show_records(train_records[:3], ncols=3, class_map=class_map)

    # Datasets
    # Transforms
    presize = 512
    size = 384
    train_tfms = tfms.A.Adapter(
        [*tfms.A.aug_tfms(size=size, presize=presize), tfms.A.Normalize()]
    )
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size), tfms.A.Normalize()])

    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    samples = [train_ds[0] for _ in range(3)]
    show_samples(samples, ncols=3, class_map=class_map, denormalize_fn=denormalize_imagenet)

    # EffecientDet Specific Part

    # DataLoaders
    train_dl = efficientdet.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
    valid_dl = efficientdet.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)
    batch, samples = first(train_dl)
    show_samples(
        samples[:6], class_map=class_map, ncols=3, denormalize_fn=denormalize_imagenet
    )

    # Model
    model = efficientdet.model(
        model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size
    )

    # Define metrics
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

    # Train using pytorch-lightning
    class LightModel(faster_rcnn.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

    light_model = LightModel(model, metrics=metrics)

    trainer = pl.Trainer(max_epochs=10, gpus=1)
    trainer.fit(light_model, train_dl, valid_dl)

    # Inference
    # DataLoader
    infer_dl = efficientdet.infer_dl(valid_ds, batch_size=8)
    # Predict
    samples, preds = efficientdet.predict_dl(model, infer_dl)
    # Show samples
    imgs = [sample["img"] for sample in samples]
    show_preds(
        imgs=imgs[:6],
        preds=preds[:6],
        class_map=class_map,
        denormalize_fn=denormalize_imagenet,
        ncols=3,
    )
    ```