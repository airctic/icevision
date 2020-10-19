# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### IMPORTANT
- **Switched from poetry to setuptools**

### Added
- Template code for `parsers.SizeMixin` if `parsers.FilepathMixin` is used
- Get image size without opening image with `get_image_size`
- Ability to skip record while parsing with `AbortParseRecord`
- Autofix for record
- Record class and mixins
- InvalidDataError for BBox
- Catches InvalidDataError while parsing data

### Changed
- **Breaking:** Unified `parsers.SizeMixin` functions `image_width` and `image_height` into a single function `image_width_height`
- Rename Parser `SizeMixin` fields from `width` `height` to `image_width` `image_height`

### Deleted

## [0.1.6]

### Added
- Efficientdet now support empty annotations

### Changed
- Returns float instead of dict on `FastaiMetricAdapter`

## [0.1.5]

### Changed

- Updates fastai2 to the final release version


## [0.1.4]

### Added

- soft import `icedata` in `icevision.all`
- `show_pbar` parameter to `COCOMetric`

### Changed

### Deleted

## [0.1.3]

### Changed
- Effdet as direct dependency

## [0.1.2]

### Added
- `show_results` function for each model

### Changed
- Default `data_splitter` for Parser changed to `RandomSplitter`
- Renamed package from `mantisshrimp` to `icevision`

### Deleted
- Removed `datasets` module to instead use the new `icedata` package

## [0.0.9]

### Added

- `batch, samples = <model_name>.build_infer_batch(dataset)`  
- `preds = <model_name>.predict(model, batch)` 
- `infer_dl = <model_name>.infer_dataloader(dataset)`
- `samples, preds = predict_dl(model, infer_dl)`
- `Dataset.from_images` Contructs a `Dataset` from a list of images (numpy arrays)
- `tfms.A.aug_tfms` for easy access to common augmentation transforms with albumentations
- `tfms.A.resize_and_pad`, useful as a validation transform
- `**predict_kwargs` to `predict_dl` signature
- `from mantisshrimp.all import *` to import internal modules and external imports
- `show` parameter to `show_img`
- `download_gdrive` and `download_and_extract_gdrive`
- New datasets `pennfundan` and `birds`

### Changed

- Renames `AlbuTransform` to `AlbumentationTransforms`
- All `build_batch` method now returns `batch, samples`, the batch is always a tuple of inputs to the model
- `batch_tfms` moved to `tfms.batch`
- `AlbumentationTransforms` moved to `tfms.A.Adapter`
- All parsers function were moved to their own namespace `parsers` instead of being on the global namespace
so, for example, instead of `Parser` now we have to do `parsers.Parser`
- Removed `Parser` word from Mixins, e.g. `ImageidParserMixin` -> `parsers.ImageidMixin`
- Removed `Parser` word from parser default bundle, e.g. `FasterRCNNParser` -> `parsers.FasterRCNN`
- COCO and VOC parsers moved from `datasets` to `parsers`
- `DataSplitter`s moved from `parsers/splits.py` to `utils/data_splitter.py`
- Renames `*_dataloader` to `*_dl`, e.g. `mask_rcnn.train_dataloader` to `mask_rcnn.train_dl`
- Moves `RecordType` from `parsers` to `core`
- Refactors `IDMap`, adds methods `get_name` and `get_id`
- Moves `IDMap` from `utils` to `data`
- `DataSplitter.split` now receives `idmap` instead of `ids`


## [0.0.0-pre-release]

### Added

- `CaptureStdout` for capturing writes to stdout (print), e.g. from COCOMetric
- `mantisshrimp.models.<model_name>.convert_raw_predictions` to convert raw preds (tensors output from the model) to library standard dict
- `COCOMetricType` for selecting what metric type to use (`bbox`, `mask`, `keypoints`)
- `COCOMetric` fixed
- `sort` parameter for `get_image_files`
- `ClassMap`: A class that handles the mapping between ids and names, with the optional insertion of the background class

### Changed

-  All dataloaders now return the batch and the records, e.g. `return (images, targets), records`
- `Metric.accumulate` signature changed to `(records, preds)`, reflects in `FastaiMetricAdapter` and `LightningModelAdapter`
- `datasets.<name>.CLASSES` substituted by a function `datasets.<name>.class_map` that returns a `ClassMap`
- `datasets.voc.VocXmlParser`, `show` methods: parameter `classes: Sequence[str]` substituted by `class_map: ClassMap`
- `datasets.fridge.parser`, `datasets.pets.parser`: additional required parameter `class_map`

### Removed
- `MantisFasterRCNN`, `MantisMaskRCNN`
- `MantisEfficientDet`
- `CategoryMap`, `Category`
- `MantisModule`


## Links  
[Unreleased]: https://github.com/airctic/mantisshrimp/tree/master
