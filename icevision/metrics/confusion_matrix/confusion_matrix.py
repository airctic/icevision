__all__ = ["SimpleConfusionMatrix"]

import sklearn, PIL
from icevision.imports import *
from icevision.metrics.metric import *
from icevision.metrics.confusion_matrix.confusion_matrix_utils import *
from pytorch_lightning import loggers as pl_loggers


class SimpleConfusionMatrix(Metric):
    def __init__(self, confidence_threshold: float = 0.001, iou_threshold: float = 0.5):
        super(SimpleConfusionMatrix, self).__init__()
        self.ground_truths = []
        self.predictions = []
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self.class_map = None
        self.confusion_matrix: sklearn.metrics.confusion_matrix = None

    def _reset(self):
        self.ground_truths = []
        self.predictions = []

    def accumulate(self, records, preds):
        if self.class_map is None:
            self.class_map = first(records).detection.class_map
        for image_targets, image_preds in zip(records, preds):
            # skip if empty ground_truths
            if not image_targets.detection.bboxes:
                continue
            targets, matched_preds = match_preds_with_targets(
                image_preds,
                image_targets,
                self._iou_threshold,
                self._confidence_threshold,
            )

            target_labels = [target_item[1] for target_item in targets]
            predicted_labels = [pred_item[0].label for pred_item in matched_preds]
            # We need to store the entire list of gts/preds to support various CM logging methods
            self.ground_truths.extend(target_labels)
            self.predictions.extend(predicted_labels)

    def finalize(self):
        """Convert preds to numpy arrays and calculate the CM"""
        assert len(self.ground_truths) == len(self.predictions)
        self.class_map = add_unknown_labels(
            self.ground_truths, self.predictions, self.class_map
        )
        # this needs to be hacked, cause it may happen that we dont have all gts/preds classes in a batch.
        # This results in missing values and class_map / gts shape mismatch
        dummy_labels = [i for i in range(self.class_map.num_classes)]
        dummy_diagonal = np.eye(self.class_map.num_classes)
        self.ground_truths = np.array(self.ground_truths + dummy_labels)
        self.predictions = np.array(self.predictions + dummy_labels)
        self.confusion_matrix = sklearn.metrics.confusion_matrix(
            y_true=self.ground_truths, y_pred=self.predictions
        )
        self.confusion_matrix = self.confusion_matrix - dummy_diagonal
        return {"dummy_value_for_fastai": -1}

    def plot(
        self,
        normalize: Optional[str] = None,
        xticks_rotation="vertical",
        values_format: str = None,
        cmap: str = "PuBu",
        figsize: int = 11,
        **display_args
    ):
        """
        A handle to plot the matrix in a jupyter notebook, potentially this could also be passed to save_fig
        """
        if normalize not in ["true", "pred", "all", None]:
            raise ValueError(
                "normalize must be one of {'true', 'pred', " "'all', None}"
            )
        # properly display ints and floats
        if values_format is not None:
            values_format = ".2f" if normalize else "d"

        cm = self._maybe_normalize(self.confusion_matrix, normalize)
        cm_display = sklearn.metrics.ConfusionMatrixDisplay(
            cm, display_labels=self.class_map._id2class
        )
        figure = cm_display.plot(
            xticks_rotation=xticks_rotation,
            cmap=cmap,
            values_format=values_format,
            **display_args
        ).figure_
        figure.set_size_inches(figsize, figsize)
        return figure

    def _fig2img(self, fig):
        """Converts matplotlib figure object to PIL Image for easier logging. Writing to buffer is necessary
        to avoid wandb cutting our labels off. Wandb autoconvert doesn't pass the `bbox_inches` parameter so we need
        to do this manually."""
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight")
        buf.seek(0)
        return PIL.Image.open(buf)

    def _maybe_normalize(self, cm, normalize):
        """This method is copied from sklearn. Only used in plot_confusion_matrix but we want to be able
        to normalize upon plotting."""
        with np.errstate(all="ignore"):
            if normalize == "true":
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif normalize == "pred":
                cm = cm / cm.sum(axis=0, keepdims=True)
            elif normalize == "all":
                cm = cm / cm.sum()
        cm = np.nan_to_num(cm)
        return cm

    def log(self, logger_object) -> None:
        if isinstance(logger_object, pl_loggers.WandbLogger):
            fig = self.plot()
            image = self._fig2img(fig)
            logger_object.experiment.log({"Confusion Matrix": wandb.Image(image)})
        self._reset()
        return
