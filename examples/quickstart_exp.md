# Training and End-to-End dataset (Fridge Objects)

In this example, we are training the Fridge Objects dataset using either [Fastai](https://github.com/fastai/fastai) or [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) training loop

=== "Fastai"
    ```python  hl_lines="36-38"
    
    # pip install icevision[all] icedata

    from icevision.all import *

    url = "https://cvbp.blob.core.windows.net/public/datasets/object_detection/odFridgeObjects.zip"
    dest_dir = "fridge"

    # Loading Data
    data_dir = icedata.load_data(url, dest_dir)

    # Parser
    class_map = ClassMap(["milk_bottle", "carton", "can", "water_bottle"])
    parser = parsers.voc(annotations_dir=data_dir / "odFridgeObjects/annotations/",
                        images_dir=data_dir / "odFridgeObjects/images",
                        class_map=class_map)

    # Records
    train_records, valid_records = parser.parse()

    # Transforms
    train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=384, presize=512), tfms.A.Normalize()])
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(384), tfms.A.Normalize()])

    # Datasets
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    # DataLoaders
    train_dl = efficientdet.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
    valid_dl = efficientdet.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

    # Model and Metrics
    model = efficientdet.model(model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size)
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

    # Training using Fastai
    learn = efficientdet.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics)
    learn.fine_tune(50, 1e-2, freeze_epochs=20)

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
    ```python  hl_lines="36-43"

    # pip install icevision[all] icedata

    from icevision.all import *
    import icedata

    url = "https://cvbp.blob.core.windows.net/public/datasets/object_detection/odFridgeObjects.zip"
    dest_dir = "fridge"

    # Loading Data
    data_dir = icedata.load_data(url, dest_dir)

    # Parser
    class_map = ClassMap(["milk_bottle", "carton", "can", "water_bottle"])
    parser = parsers.voc(annotations_dir=data_dir / "odFridgeObjects/images/",
                        images_dir=data_dir / "odFridgeObjects/annotations",
                        class_map=class_map)

    # Records
    train_records, valid_records = parser.parse()

    train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=384, presize=512), tfms.A.Normalize()])
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(384), tfms.A.Normalize()])

    # Datasets
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    # DataLoaders
    train_dl = efficientdet.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
    valid_dl = efficientdet.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

    # Model and Metrics
    model = efficientdet.model(model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size)
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

    # Training using Pytorch Lightning
    class LightModel(efficientdet.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-2)    

    light_model = LightModel(model, metrics=metrics)
    trainer = pl.Trainer(max_epochs=70, gpus=1)
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
