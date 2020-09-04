# Models

[**Source**](https://github.com/airctic/icevision/tree/master/icevision/models)

IceVision offers both [EffecientDet](https://airctic.github.io/icevision/model_efficientdet/) and [Faster RCNN](https://airctic.github.io/icevision/model_faster_rcnn/) models.

IceVision uses a unified API that makes it easy for the users to swap one model by another as it is shown in the following example. Click one or the other tab to compare both implementations and discover the strong similarities between the two implementations:

=== "EffecientDet"
    ```python  hl_lines="33 34 37 43 51 54"
    # Installing IceVision
    # !pip install git+git://github.com/airctic/icevision.git#egg=icevision[all] --upgrade

    # Imports
    from icevision.all import *

    # Common part to all models

    # Loading Data
    data_dir = datasets.pets.load()

    # Parser
    class_map = datasets.pets.class_map()
    parser = datasets.pets.parser(data_dir, class_map)
    data_splitter = RandomSplitter([0.8, 0.2])
    train_records, valid_records = parser.parse(data_splitter)
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

    # EffecientDet Specific Part
    # DataLoaders
    train_dl = efficientdet.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
    valid_dl = efficientdet.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

    # Model
    model = efficientdet.model(
        model_name="tf_efficientdet_lite0", num_classes=len(class_map), img_size=size
    )

    # Fastai Learner
    metrics = [COCOMetric()]
    learn = efficientdet.fastai.learner(
        dls=[train_dl, valid_dl], model=model, metrics=metrics
    )

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



=== "Faster RCNN"
    ```python hl_lines="33 34 37 41 49 52"
    # Installing IceVision
    # !pip install git+git://github.com/airctic/icevision.git#egg=icevision[all] --upgrade

    # Imports
    from icevision.all import *

    # Common part to all models

    # Loading Data
    data_dir = datasets.pets.load()

    # Parser
    class_map = datasets.pets.class_map()
    parser = datasets.pets.parser(data_dir, class_map)
    data_splitter = RandomSplitter([0.8, 0.2])
    train_records, valid_records = parser.parse(data_splitter)
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

    # Faster RCNN Specific Part
    # DataLoaders
    train_dl = faster_rcnn.train_dl(train_ds, batch_size=16, num_workers=4, shuffle=True)
    valid_dl = faster_rcnn.valid_dl(valid_ds, batch_size=16, num_workers=4, shuffle=False)

    # Model
    model = faster_rcnn.model(num_classes=len(class_map))

    # Fastai Learner
    metrics = [COCOMetric()]
    learn = faster_rcnn.fastai.learner(
        dls=[train_dl, valid_dl], model=model, metrics=metrics
    )

    learn.fine_tune(10, 1e-2, freeze_epochs=1)

    # Inference
    # DataLoader
    infer_dl = faster_rcnn.infer_dl(valid_ds, batch_size=8)
    
    # Predict
    samples, preds = faster_rcnn.predict_dl(model, infer_dl)
    
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
