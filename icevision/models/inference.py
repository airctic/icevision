__all__ = ["process_bbox_predictions", "_end2end_detect"]

from icevision.imports import *
from icevision.core import *
from icevision.data import *
from icevision.tfms.albumentations.albumentations_helpers import (
    get_size_without_padding,
)
from icevision.tfms.albumentations import albumentations_adapter

from icevision.utils.imageio import *
from icevision.visualize.draw_data import *


def _end2end_detect(
    img: Union[PIL.Image.Image, Path, str],
    transforms: albumentations_adapter.Adapter,
    model: torch.nn.Module,
    class_map: ClassMap,
    detection_threshold: float = 0.5,
    predict_fn: Callable = None,
):
    """
    Run Object Detection inference (only `bboxes`) on a single image.

    Parameters
    ----------
    img: image to run inference on. Can be a string, Path or PIL.Image
    transforms: icevision albumentations transforms
    model: model to run inference with
    class_map: ClassMap with the available categories
    detection_threshold: confidence threshold below which boxes are discarded

    Returns
    -------
    List of dicts with category, score and bbox coordinates adjusted to original image size and aspect ratio
    """
    if isinstance(img, (str, Path)):
        img = PIL.Image.open(Path(img))

    infer_ds = Dataset.from_images([np.array(img)], transforms, class_map=class_map)
    pred = predict_fn(model, infer_ds, detection_threshold=detection_threshold)[0]
    bboxes = process_bbox_predictions(pred, img, transforms.tfms_list)
    return bboxes


def process_bbox_predictions(
    pred: Prediction,
    img: PIL.Image.Image,
    transforms: List[Any],
) -> List[Dict[str, Any]]:
    """
    Postprocess prediction.

    Parameters
    ----------
    pred: icevision prediction object
    img: original image, before any model-pre-processing done
    transforms: list of model-pre-processing transforms

    Returns
    -------
    List of dicts with class, score and bbox coordinates
    """
    bboxes = []
    for bbox, score, label in zip(
        pred.pred.detection.bboxes,
        pred.pred.detection.scores,
        pred.pred.detection.labels,
    ):
        xmin, ymin, xmax, ymax = postprocess_bbox(
            img, bbox, transforms, pred.pred.height, pred.pred.width
        )
        result = {
            "class": label,
            "score": score,
            "bbox": [xmin, ymin, xmax, ymax],
        }
        bboxes.append(result)
    return bboxes


def postprocess_bbox(
    img: PIL.Image.Image, bbox: BBox, transforms: List[Any], h_after: int, w_after: int
) -> Tuple[int, int, int, int]:
    """
    Post-process predicted bbox to adjust coordinates to input image size.

    Parameters
    ----------
    img: original image, before any model-pre-processing done
    bbox: predicted bbox
    transforms: list of model-pre-processing transforms
    h_after: height of image after model-pre-processing transforms
    w_after: width of image after model-pre-processing transforms

    Returns
    -------
    Tuple with (xmin, ymin, xmax, ymax) rescaled and re-adjusted to match the original image size
    """
    w_before, h_before = img.size
    h_after, w_after = get_size_without_padding(transforms, img, h_after, w_after)
    pad = np.abs(h_after - w_after) // 2

    h_scale, w_scale = h_after / h_before, w_after / w_before
    if h_after < w_after:
        xmin, xmax, ymin, ymax = (
            int(bbox.xmin),
            int(bbox.xmax),
            int(bbox.ymin) - pad,
            int(bbox.ymax) - pad,
        )
    else:
        xmin, xmax, ymin, ymax = (
            int(bbox.xmin) - pad,
            int(bbox.xmax) - pad,
            int(bbox.ymin),
            int(bbox.ymax),
        )

    xmin, xmax, ymin, ymax = (
        max(xmin, 0),
        min(xmax, w_after),
        max(ymin, 0),
        min(ymax, h_after),
    )
    xmin, xmax, ymin, ymax = (
        int(xmin / w_scale),
        int(xmax / w_scale),
        int(ymin / h_scale),
        int(ymax / h_scale),
    )

    return xmin, ymin, xmax, ymax


def draw_img_and_boxes(
    img: Union[PIL.Image.Image, np.ndarray],
    bboxes: dict,
    class_map,
    display_score: bool = True,
    label_color: Union[np.array, list, tuple, str] = (255, 255, 0), 
    label_border_color: Union[np.array, list, tuple, str] = (255, 255, 0), 
    ) -> PIL.Image.Image:
    
    if not isinstance(img, PIL.Image.Image):
      img = np.array(img)

    # convert dict to record
    record = ObjectDetectionRecord()
    w, h = img.shape
    record.img = np.array(img)
    record.set_img_size(ImgSize(width=w, height=h))
    record.detection.set_class_map(class_map)
    for bbox in bboxes:
        record.detection.add_bboxes([BBox.from_xyxy(*bbox['bbox'])])
        if display_score== True:
          score = bbox['score']
          score = f"{score * 100: .2f}%"
          label = bbox['class']
          # label = f"{label}: {score}"
          record.detection.add_labels([label])
        else:
          record.detection.add_labels([bbox['class']])
        

    pred_img = draw_sample(record, label_color=label_color, label_border_color=label_border_color)

    return pred_img
