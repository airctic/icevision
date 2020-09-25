# Using Mask RCNN

This shows how to train a MaskRCNN model on  the [Penn-Fundan](https://www.cis.upenn.edu/~jshi/ped_html/) dataset using either [Fastai](https://github.com/fastai/fastai) or [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) training loop.

=== "Fastai"
    ```python  hl_lines="53-55"

    # Install icevision
    # !pip install icevision[all] icedata

    # Import everything from icevision
    from icevision.all import *
    import icedata

    # Load the data and create the parser
    data_dir = icedata.pennfudan.load_data()
    class_map = icedata.pennfudan.class_map()
    parser = icedata.pennfudan.parser(data_dir)

    # Parse records with random splits
    train_records, valid_records = parser.parse()

    # Define the transforms and create the Datasets
    presize = 512
    size = 384
    shift_scale_rotate = tfms.A.ShiftScaleRotate(rotate_limit=10)
    crop_fn = partial(tfms.A.RandomSizedCrop, min_max_height=(size // 2, size), p=0.5)
    train_tfms = tfms.A.Adapter(
        [
            *tfms.A.aug_tfms(
                size=size,
                presize=presize,
                shift_scale_rotate=shift_scale_rotate,
                crop_fn=crop_fn,
            ),
            tfms.A.Normalize(),
        ]
    )
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    # Shows how the transforms affects a single sample
    samples = [train_ds[0] for _ in range(6)]
    show_samples(
        samples, denormalize_fn=denormalize_imagenet, ncols=3, label=False, show=True
    )

    # Create DataLoaders
    train_dl = mask_rcnn.train_dl(train_ds, batch_size=16, shuffle=True, num_workers=4)
    valid_dl = mask_rcnn.valid_dl(valid_ds, batch_size=16, shuffle=False, num_workers=4)

    # Define metrics for the model
    # TODO: Currently broken for Mask RCNN
    # metrics = [COCOMetric(COCOMetricType.mask)]

    # Create model
    model = mask_rcnn.model(num_classes=len(class_map))

    # Create Fastai Learner and train the model
    learn = mask_rcnn.fastai.learner(model=model)
    learn.fine_tune(10, 5e-4, freeze_epochs=2)

    # BONUS: Use model for inference. In this case, let's take some images from valid_ds
    # Take a look at `Dataset.from_images` if you want to predict from images in memory
    samples = [valid_ds[i] for i in range(6)]
    batch, samples = mask_rcnn.build_infer_batch(samples)
    preds = mask_rcnn.predict(model=model, batch=batch)

    imgs = [sample["img"] for sample in samples]
    show_preds(imgs=imgs, preds=preds, denormalize_fn=denormalize_imagenet, ncols=3)
    ```

=== "Pytorch Lightning"
    ```python  hl_lines="53-61"

    # Install icevision
    # !pip install icevision[all] icedata

    # Import everything from icevision
    from icevision.all import *
    import icedata

    # Load the data and create the parser
    data_dir = icedata.pennfudan.load_data()
    class_map = icedata.pennfudan.class_map()
    parser = icedata.pennfudan.parser(data_dir)


    train_records, valid_records = parser.parse()

    # Define the transforms and create the Datasets
    presize = 512
    size = 384
    shift_scale_rotate = tfms.A.ShiftScaleRotate(rotate_limit=10)
    crop_fn = partial(tfms.A.RandomSizedCrop, min_max_height=(size // 2, size), p=0.5)
    train_tfms = tfms.A.Adapter(
        [
            *tfms.A.aug_tfms(
                size=size,
                presize=presize,
                shift_scale_rotate=shift_scale_rotate,
                crop_fn=crop_fn,
            ),
            tfms.A.Normalize(),
        ]
    )
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    # Shows how the transforms affects a single sample
    samples = [train_ds[0] for _ in range(6)]
    show_samples(
        samples, denormalize_fn=denormalize_imagenet, ncols=3, label=False, show=True
    )

    # Create DataLoaders
    train_dl = mask_rcnn.train_dl(train_ds, batch_size=16, shuffle=True, num_workers=4)
    valid_dl = mask_rcnn.valid_dl(valid_ds, batch_size=16, shuffle=False, num_workers=4)

    # Define metrics for the model
    # TODO: Currently broken for Mask RCNN
    # metrics = [COCOMetric(COCOMetricType.mask)]

    # Create model
    model = mask_rcnn.model(num_classes=len(class_map))

    # Train using pytorch-lightning
    class LightModel(faster_rcnn.lightning.ModelAdapter):
        def configure_optimizers(self):
            return SGD(self.parameters(), lr=1e-4)

    light_model = LightModel(model, metrics=metrics)

    trainer = pl.Trainer(max_epochs=10, gpus=1)
    trainer.fit(light_model, train_dl, valid_dl)

    # BONUS: Use model for inference. In this case, let's take some images from valid_ds
    # Take a look at `Dataset.from_images` if you want to predict from images in memory
    samples = [valid_ds[i] for i in range(6)]
    batch, samples = mask_rcnn.build_infer_batch(samples)
    preds = mask_rcnn.predict(model=model, batch=batch)

    imgs = [sample["img"] for sample in samples]
    show_preds(imgs=imgs, preds=preds, denormalize_fn=denormalize_imagenet, ncols=3)
    ```