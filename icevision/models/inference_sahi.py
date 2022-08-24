__all__ = ["IceSahiModel"]

from types import ModuleType
from icevision.imports import *
from icevision.imports import *
from icevision.core import *
from icevision.data import *
from icevision.utils.imageio import *
from icevision.visualize.draw_data import *
from icevision.visualize.utils import *
from icevision.tfms.albumentations import albumentations_adapter


from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.predict import get_sliced_prediction as sahi_get_sliced_prediction


class IceSahiModel(DetectionModel):
    def __init__(
        self,
        model_type: ModuleType,
        model: torch.nn.Module,
        class_map: ClassMap,
        tfms: albumentations_adapter.Adapter,
        confidence_threshold: float = 0.5,
    ):
        super().__init__(
            model_path="", confidence_threshold=confidence_threshold, load_at_init=False
        )

        self.model_type = model_type
        self.class_map = class_map
        self.confidence_threshold = confidence_threshold
        self.model = model
        self.tfms = tfms
        self.category_mapping = self.class_map._class2id

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
            image_size: int
                Inference input size.
        """
        self._original_predictions = self.model_type.end2end_detect(
            img=PIL.Image.fromarray(image),
            transforms=self.tfms,
            model=self.model,
            class_map=self.class_map,
            detection_threshold=self.confidence_threshold,
        )

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return self.class_map.num_classes

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return False

    @property
    def category_names(self):
        return self.class_map.get_classes()

    def get_sliced_prediction(
        self,
        image: Union[PIL.Image.Image, np.ndarray, Path, str],
        keep_sahi_format: bool = False,
        display_label: bool = True,
        display_bbox: bool = True,
        display_score: bool = True,
        font_path: Optional[os.PathLike] = get_default_font(),
        font_size: Union[int, float] = 10,
        return_as_pil_img=True,
        return_img=True,
        **kwargs
    ):
        if isinstance(image, Path):
            image = str(image)

        pred = sahi_get_sliced_prediction(image=image, detection_model=self, **kwargs)
        if keep_sahi_format:
            return pred
        else:
            scores = []
            label_ids = []
            bboxes = []
            record = BaseRecord(
                (
                    BBoxesRecordComponent(),
                    InstancesLabelsRecordComponent(),
                    ScoresRecordComponent(),
                    ImageRecordComponent(),
                )
            )

            for pred in pred.object_prediction_list:
                scores.append(pred.score.value)
                label_ids.append(pred.category.name)
                bboxes.append(
                    BBox.from_xyxy(
                        pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy
                    )
                )

            record.detection.set_class_map(self.class_map)
            record.detection.add_labels(label_ids)
            record.detection.add_bboxes(bboxes)
            record.detection.set_scores(np.array(scores))

            img_path = str(image)
            if isinstance(image, (str, Path)):
                image = open_img(img_path, ensure_no_data_convert=True)

            record.set_img(image)

            img_size = get_img_size(img_path)

            if return_img:
                pred_img = draw_record(
                    record=record,
                    class_map=self.class_map,
                    display_label=display_label,
                    display_score=display_score,
                    display_bbox=display_bbox,
                    font_path=font_path,
                    font_size=font_size,
                    return_as_pil_img=return_as_pil_img,
                )
            else:
                record._unload()

            pred_dict = record.as_dict()

            if return_img:
                pred_dict["img"] = pred_img
            else:
                pred_dict["img"] = None

            pred_dict["width"] = img_size.width
            pred_dict["height"] = img_size.height

            del pred_dict["common"]

            return pred_dict

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions["detection"]
        category_mapping = self.category_mapping

        categories = list(category_mapping.keys())
        boxes = {v: [] for v in category_mapping.values() if v != 0}
        labels, scores = [], []

        total_detections = len(original_predictions["labels"])

        # compatilibty for sahi v0.8.15
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        # assuming self._original_predictions["detection"] contains single image results
        image_ind = 0
        shift_amount = shift_amount_list[image_ind]
        full_shape = None if full_shape_list is None else full_shape_list[image_ind]

        for i in range(total_detections):
            lbl = original_predictions["label_ids"][i]
            scores.append(original_predictions["scores"][i])
            boxes[lbl].append(original_predictions["bboxes"][i].xyxy)
            labels.append(lbl)

        object_prediction_list = []

        for category_id in list(category_mapping.values())[1:]:
            category_boxes = boxes[category_id]
            num_category_predictions = len(category_boxes)

            for category_predictions_ind in range(num_category_predictions):
                bbox = category_boxes[category_predictions_ind]
                score = scores[category_predictions_ind]
                bool_mask = None

                category_name = categories[category_id]
                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=bool_mask,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)

        self._object_prediction_list_per_image = [object_prediction_list]
