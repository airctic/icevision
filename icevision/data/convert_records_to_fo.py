__all__ = [
    "convert_record_to_fo_sample",
    "convert_prediction_to_fo_sample",
    "create_fo_dataset",
]

from PIL import Image
from warnings import warn
from pathlib import Path

from icevision import tfms
from icevision.core.bbox import BBox
from icevision.data.prediction import Prediction
from icevision.models.inference import postprocess_bbox
from icevision.soft_dependencies import SoftDependencies
from icevision.utils.imageio import get_img_size

from typing import Any, Callable, Union, Iterable, List, Tuple

if SoftDependencies.fiftyone:
    from fiftyone import (
        Detection,
        Detections,
        Sample,
        Dataset,
        list_datasets,
        load_dataset,
    )

    fiftyone_available = True
else:
    fiftyone_available = False


def convert_record_to_fo_sample(
    record,
    field_name: str,
    sample: Union[None, Sample] = None,
    autosave: bool = False,
    transformations=None,  # Undo postprocess via sample: icevision.models.inference.postprocess_bbox
    undo_bbox_tfms_fn: Callable[[BBox], BBox] = None,  # Undo postprocess custom fn
    filepath: Union[Path, str] = None,
) -> Sample:
    """
    Takes a record and applies it to an fiftyone sample.

    A record contains a preprocessed image of you data. However, fiftyone expects an filepath to you image. Thus, we need to undo the pre-processing
    of the bounding box. Therefore, we have provide 2 methods

        - The postprocess_bbox() function form icevision.models.inference

        - A custom undo_bbox_tfms_fn


    If sample is not defined, a new sample is created.

    Parameters
    ----------
    record: An icevision record
    field_name: A fo field name for the sample
    sample: If no sample is given, a sample is created
    autosave: Whether an existing sample should be saved directly

    transformations: list of model-pre-processing transforms

    undo_bbox_tfms_fn: Custom function, that undoes transformations

    filepath: The filepath, if it is not present in the record

    Returns
    -------
    fo.Sample, which can be directly added to a given Fiftyone dataset
    """

    if not fiftyone_available:
        raise ImportError("Fiftyone is not installed on you system.")

    # Validate input
    if transformations is not None and undo_bbox_tfms_fn is not None:
        warn(
            "Both transformations and undo_bbox_tfms_fn is defined. You custom function will be used."
        )
    if transformations is None and undo_bbox_tfms_fn is None:
        warn(
            "No bounding box postprocessing is used. The bboxes might be shifted in fiftyone"
        )

    if filepath is not None:
        _internal_filepath = filepath
    else:
        if sample is not None:
            _internal_filepath = sample.filepath
        else:
            _internal_filepath = record.common.filepath

    # Prepare undo bbox tfms fn
    img_size = get_img_size(_internal_filepath)
    if undo_bbox_tfms_fn is not None:
        _internal_undo_bbox_tfms = lambda bbox: _convert_bbox_to_fo_bbox(
            undo_bbox_tfms_fn(bbox), img_size.width, img_size.height
        )
    elif transformations is not None:
        _internal_undo_bbox_tfms = lambda bbox: _convert_bbox_to_fo_bbox(
            postprocess_bbox(
                img_size,
                bbox,
                transformations,
                record.common.width,
                record.common.height,
            ),
            img_size.width,
            img_size.height,
        )
    else:
        _internal_undo_bbox_tfms = lambda bbox: _convert_bbox_to_fo_bbox(
            bbox, img_size.width, img_size.height
        )

    # Get fo.Detections
    detections = record_to_fo_detections(record, _internal_undo_bbox_tfms)

    # Get sample after successful detection
    if sample is None:
        sample = Sample(_internal_filepath)
    elif isinstance(sample, Sample):
        pass
    else:
        raise ValueError(f"Sample {sample} is not None or fo.Sample")

    # Add detections to sample
    sample[field_name] = Detections(detections=detections)

    if autosave:
        sample.save()

    return sample


def record_to_fo_detections(record, undo_bbox_tfms) -> Iterable[Detection]:

    if hasattr(record, "detection"):
        if hasattr(record.detection, "bboxes"):
            if hasattr(record.detection, "scores"):
                detections = [
                    Detection(
                        label=label, bounding_box=undo_bbox_tfms(bbox), confidence=score
                    )
                    for label, bbox, score in zip(
                        record.detection.labels,
                        record.detection.bboxes,
                        record.detection.scores,
                    )
                ]
            else:
                detections = [
                    Detection(label=label, bounding_box=undo_bbox_tfms(bbox))
                    for label, bbox in zip(
                        record.detection.labels, record.detection.bboxes
                    )
                ]
        else:
            raise NotImplementedError("Fiftyone export support only bboxes yet")
    else:
        raise ValueError(f"Provided record does not contain a detection attribute")

    return detections


def convert_prediction_to_fo_sample(
    prediction: Prediction,
    gt_field_name="ground_truth",
    pred_field_name="prediction",
    transformations=None,  # Undo postprocess via sample: icevision.models.inference.postprocess_bbox
    undo_bbox_tfms_fn: Callable[
        [Union[List[int], BBox]], List[int]
    ] = None,  # Undo postprocess custom fn
) -> Sample:
    """
    Takes a prediction object and created a fiftyone sample.

    A record contains a preprocessed image of you data. However, fiftyone expects an filepath to you image. Thus, we need to undo the pre-processing
    of the bounding box. Therefore, we have provide 2 methods

        - The postprocess_bbox() function form icevision.models.inference

        - A custom undo_bbox_tfms_fn

    If sample is not defined, a new sample is created.

    Parameters
    ----------
    record: An icevision record
    gt_field_name: Name of the ground truth field in the sample
    pred_field_name: Name of the prediction field in the sample

    transformations: list of model-pre-processing transforms

    undo_bbox_tfms_fn: Custom function, that undoes transformations

    Returns
    -------
    fo.Sample, which can be directly added to a given Fiftyone dataset
    """

    if not fiftyone_available:
        raise ImportError("Fiftyone is not installed on you system.")

    # Create fo sample
    sample = Sample(prediction.ground_truth.common.filepath)

    # Add gt
    convert_record_to_fo_sample(
        prediction.ground_truth,
        gt_field_name,
        sample,
        False,
        transformations,
        undo_bbox_tfms_fn,
    )

    # Add prediction
    convert_record_to_fo_sample(
        prediction.pred,
        pred_field_name,
        sample,
        False,
        transformations,
        undo_bbox_tfms_fn,
        filepath=prediction.ground_truth.common.filepath,
    )

    return sample


def create_fo_dataset(
    detections: List[Any],
    dataset_name: str,
    exist_ok: bool = False,
    persistent=False,
    field_name: str = "icevision_record",
    transformations=None,  # Undo postprocess via sample: icevision.models.inference.postprocess_bbox
    undo_bbox_tfms_fn: Callable[
        [Union[List[int], BBox]], List[int]
    ] = None,  # Undo postprocess custom fn
) -> Dataset:
    """
    Takes an iter of either Predictions or records and created or adds them to a fo.Dataset

    A record contains a preprocessed image of you data. However, fiftyone expects an filepath to you image. Thus, we need to undo the pre-processing
    of the bounding box. Therefore, we have provide 2 methods

        - The postprocess_bbox() function form icevision.models.inference

        - A custom undo_bbox_tfms_fn

    If sample is not defined, a new sample is created.

    Parameters
    ----------
    detections: An iterable of iv Predictions or records
    dataset_name: Name of dataset
    exist_ok: Whether the data should be added to an existing dataset (active Opt-in)
    persistent:  Whether a dataset should be persistent on creation
    field_name: The field name that is provided if records are in the iterable

    transformations: list of model-pre-processing transforms
    undo_bbox_tfms_fn: Custom function, that undoes transformations

    Returns
    -------
    fo.Dataset
    """

    if not fiftyone_available:
        raise ImportError("Fiftyone is not installed on you system.")

    # Validate input
    if dataset_name in list_datasets():
        if exist_ok:
            _internal_dataset = load_dataset(dataset_name)
        else:
            raise ValueError(
                f"Dataset with name {dataset_name} already exist and exist_ok is {exist_ok}"
            )
    else:
        _internal_dataset = Dataset(name=dataset_name, persistent=persistent)

    _sample_list = []
    for element in detections:
        if isinstance(element, Prediction):
            _sample_list.append(
                convert_prediction_to_fo_sample(
                    element,
                    transformations=transformations,
                    undo_bbox_tfms_fn=undo_bbox_tfms_fn,
                )
            )
        else:
            _sample_list.append(
                convert_record_to_fo_sample(
                    element,
                    field_name=field_name,
                    transformations=transformations,
                    undo_bbox_tfms_fn=undo_bbox_tfms_fn,
                )
            )

    _internal_dataset.add_samples(_sample_list)

    return _internal_dataset

    # utils


def _convert_bbox_to_fo_bbox(
    bbox: Union[BBox, List, Tuple], original_width: int, original_height: int
) -> List[float]:
    if not isinstance(bbox, BBox):
        bbox = BBox(*bbox)

    return [
        bbox.xmin / original_width,
        bbox.ymin / original_height,
        bbox.width / original_width,
        bbox.height / original_height,
    ]
