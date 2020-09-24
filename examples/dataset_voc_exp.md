# How to train a VOC compatible dataset


This notebook shows a special use case of training a VOC compatible dataset using the predefined [VOC parser](https://github.com/airctic/icevision/blob/master/icevision/parsers/voc_parser.py) without creating data, and parsers files as opposed to the [fridge dataset](https://github.com/airctic/icevision/tree/master/icevision/datasets/fridge) example.

=== "Fastai"
    ```python  hl_lines="68-76"

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
    show_records(train_records[:3], ncols=3, class_map=class_map)

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

    samples = [train_ds[0] for _ in range(3)]
    show_samples(samples, ncols=3, class_map=class_map, denormalize_fn=denormalize_imagenet)

    # EffecientDet Specific Part

    # DataLoaders
    train_dl = efficientdet.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
    valid_dl = efficientdet.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

    # Show some image samples
    samples = [train_ds[5] for _ in range(3)]
    show_samples(samples, class_map=class_map, denormalize_fn=denormalize_imagenet, ncols=3)

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
    # Learning Rate Finder
    learn.freeze()
    learn.lr_find()

    # Fine tune: 2 Phases
    # Phase 1: Train the head for 10 epochs while freezing the body
    # Phase 2: Train both the body and the head during 50 epochs
    learn.fine_tune(50, 1e-2, freeze_epochs=10)

    # Inference
    infer_dl = efficientdet.infer_dl(valid_ds, batch_size=16)
    # Predict
    samples, preds = efficientdet.predict_dl(model, infer_dl)

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
    ```python  hl_lines="69-77"

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
    show_records(train_records[:3], ncols=3, class_map=class_map)

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

    samples = [train_ds[0] for _ in range(3)]
    show_samples(samples, ncols=3, class_map=class_map, denormalize_fn=denormalize_imagenet)

    # EffecientDet Specific Part

    # DataLoaders
    train_dl = efficientdet.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
    valid_dl = efficientdet.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

    # Show some image samples
    samples = [train_ds[5] for _ in range(3)]
    show_samples(samples, class_map=class_map, denormalize_fn=denormalize_imagenet, ncols=3)

    # Model
    model = efficientdet.model(
        model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size
    )

    # Define metrics
    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]

    # Fastai Learner
    metrics = [COCOMetric()]
    learn = efficientdet.fastai.learner(
        dls=[train_dl, valid_dl], model=model, metrics=metrics
    )

    # Train using pytorch-lightning
    class LightModel(faster_rcnn.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

    light_model = LightModel(model, metrics=metrics)

    trainer = pl.Trainer(max_epochs=60, gpus=1)
    trainer.fit(light_model, train_dl, valid_dl)

    # Inference
    infer_dl = efficientdet.infer_dl(valid_ds, batch_size=16)
    # Predict
    samples, preds = efficientdet.predict_dl(model, infer_dl)

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