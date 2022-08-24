__all__ = ["process_bbox_predictions", "_end2end_detect", "draw_img_and_boxes"]

from icevision.imports import *
from icevision.core import *
from icevision.data import *
from icevision.tfms.albumentations.albumentations_helpers import (
    get_size_without_padding,
)
from icevision.tfms.albumentations import albumentations_adapter

from icevision.utils.imageio import *
from icevision.visualize.draw_data import *
from icevision.visualize.utils import *
from icevision.utils.imageio import get_img_size_from_data

DEFAULT_FONT_PATH = get_default_font()


def _end2end_detect(
    img: Union[PIL.Image.Image, np.ndarray, Path, str],
    transforms: albumentations_adapter.Adapter,
    model: torch.nn.Module,
    class_map: ClassMap,
    detection_threshold: float = 0.5,
    predict_fn: Callable = None,
    display_label: bool = True,
    display_bbox: bool = True,
    display_score: bool = True,
    font_path: Optional[os.PathLike] = DEFAULT_FONT_PATH,
    font_size: Union[int, float] = 12,
    label_color: Union[np.array, list, tuple, str] = ("#FF59D6"),  # Pink
    return_as_pil_img=True,
    return_img=True,
    **kwargs,
):
    """
    Run Object Detection inference (only `bboxes`) on a single image.

    Parameters
    ----------
    img: image to run inference on. It can be a string, Path or PIL.Image or numpy
    transforms: icevision albumentations transforms
    model: model to run inference with
    class_map: ClassMap with the available categories
    detection_threshold: confidence threshold below which bounding boxes are discarded
    display_label: display or not a bounding box label(i.e class)
    display_bbox: display or not a bounding box
    display_score: display or not a bounding box score
    font_path: path to the font file used
    font_size: font size
    label_color: a <collection> of RGB values or a hex code string that defines
                   the color of all the plotted labels
    return_as_pil_img: if True a PIL image is returned otherwise a numpy array is returned
    return_img: whether we should also return an image in addition to the bounding boxes, labels, and scores

    Returns
    -------
    A dictionnary with categories, scores, bounding box coordinates, image height and width,
                   and optionally a PIL Image or a numpy array (image).
                   Bounding boxes are adjusted to the original image size and aspect ratio
    """
    if isinstance(img, (str, Path)):
        img = open_img(str(img), ensure_no_data_convert=True)

    infer_ds = Dataset.from_images([np.array(img)], transforms, class_map=class_map)
    pred = predict_fn(model, infer_ds, detection_threshold=detection_threshold)[0]
    pred = process_bbox_predictions(pred, img, transforms.tfms_list)
    record = pred.pred

    # draw image, return it as PIL image by default otherwise as numpy array
    if return_img:
        pred_img = draw_record(
            record=pred,
            class_map=class_map,
            display_label=display_label,
            display_score=display_score,
            display_bbox=display_bbox,
            font_path=font_path,
            font_size=font_size,
            label_color=label_color,
            return_as_pil_img=return_as_pil_img,
            **kwargs,
        )
        record.set_img(pred_img)
    else:
        record._unload()

    img_size = get_img_size_from_data(img)
    record.set_img_size(img_size)

    pred_dict = record.as_dict()

    # expose img at the root instead of having under the `common` key
    if return_img:
        pred_dict["img"] = pred_img
    else:
        pred_dict["img"] = None

    pred_dict["width"] = img_size.width
    pred_dict["height"] = img_size.height
    # delete the `common` key that holds both the `img` and its shape
    del pred_dict["common"]

    # return a dict that contains the image with its predicted boxes (i.e. with resized boxes that match the original image size)
    # labels, and prediction scores
    return pred_dict


def process_bbox_predictions(
    pred: Prediction,
    img: Union[PIL.Image.Image, np.ndarray],
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
    img = np.array(img)
    bboxes = []
    for bbox, score, label in zip(
        pred.pred.detection.bboxes,
        pred.pred.detection.scores,
        pred.pred.detection.labels,
    ):
        img_size = get_img_size_from_data(img)
        xmin, ymin, xmax, ymax = postprocess_bbox(
            img_size, bbox, transforms, pred.pred.height, pred.pred.width
        )

        bbox = BBox.from_xyxy(xmin, ymin, xmax, ymax)
        bboxes.append(bbox)

    pred.pred.img = np.array(img)
    pred.pred.detection.set_bboxes(bboxes)
    return pred


def postprocess_bbox(
    img_size: ImgSize, bbox: BBox, transforms: List[Any], h_after: int, w_after: int
) -> Tuple[int, int, int, int]:
    """
    Post-process predicted bbox to adjust coordinates to input image size.

    Parameters
    ----------
    img_size: original image size, before any model-pre-processing done
    bbox: predicted bbox
    transforms: list of model-pre-processing transforms
    h_after: height of image after model-pre-processing transforms
    w_after: width of image after model-pre-processing transforms

    Returns
    -------
    Tuple with (xmin, ymin, xmax, ymax) rescaled and re-adjusted to match the original image size
    """
    h_after, w_after = get_size_without_padding(transforms, img_size, h_after, w_after)
    pad = np.abs(h_after - w_after) // 2

    h_scale, w_scale = h_after / img_size.height, w_after / img_size.width
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

    img = np.array(img)

    # convert dict to record
    record = ObjectDetectionRecord()
    record.img = img

    img_size = get_img_size_from_data(img)
    record.set_img_size(img_size)

    record.detection.set_class_map(class_map)

    for bbox in bboxes:
        record.detection.add_bboxes([BBox.from_xyxy(*bbox["bbox"])])
        record.detection.add_labels([bbox["class"]])
        record.detection.set_scores(bbox["score"])

    pred_img = draw_sample(
        record,
        display_score=display_score,
        label_color=label_color,
        label_border_color=label_border_color,
    )

    return pred_img
