# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

## Links  
[Unreleased]: https://github.com/lgvaz/mantisshrimp/tree/master
