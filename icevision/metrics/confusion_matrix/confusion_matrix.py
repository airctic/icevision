__all__ = ["SimpleConfusionMatrix"]

import sklearn, PIL
from icevision.imports import *
from icevision.metrics.metric import *
from icevision.metrics.confusion_matrix.cm_utils import *
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
            self.class_map = next(iter(records))["class_map"]
        for image_targets, image_preds in zip(records, preds):
            target_bboxes = image_targets["bboxes"]
            target_labels = image_targets["labels"]
            # skip if empty ground_truths
            if not target_bboxes:
                continue
            predicted_bboxes = [
                DetectedBBox(*bbox.xyxy, score=score, label=label)
                for bbox, score, label in zip(
                    image_preds["bboxes"], image_preds["scores"], image_preds["labels"]
                )
            ]
            # get torchvision iou scores (requires conversion to tensors)
            iou_scores = pairwise_iou(predicted_bboxes, target_bboxes)
            # TODO: see what happens if that_match is empty
            that_match = torch.any(iou_scores > self._iou_threshold, dim=1)
            iou_scores = iou_scores[that_match]
            iou_scores = zeroify_items_below_threshold(
                iou_scores, threshold=self._iou_threshold
            )

            # need to use compress cause list indexing with boolean tensor isn't supported
            predicted_bboxes = list(itertools.compress(predicted_bboxes, that_match))
            predicted_bboxes = couple_with_targets(predicted_bboxes, iou_scores)
            predicted_labels = pick_best_score_labels(
                predicted_bboxes, confidence_threshold=self._confidence_threshold
            )

            assert len(predicted_labels) == len(target_labels)
            # We need to store the entire list of gts/preds to support various CM logging methods
            self.ground_truths.extend(target_labels)
            self.predictions.extend(predicted_labels)

    def finalize(self):
        """Convert preds to numpy arrays and calculate the CM"""
        assert len(self.ground_truths) == len(self.predictions)
        self.class_map = add_unknown_labels(
            self.ground_truths, self.predictions, self.class_map
        )
        self.ground_truths = np.array(self.ground_truths)
        self.predictions = np.array(self.predictions)
        self.confusion_matrix = sklearn.metrics.confusion_matrix(
            y_true=self.ground_truths, y_pred=self.predictions
        )
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
        cm = self.confusion_matrix
        with np.errstate(all="ignore"):
            if normalize == "true":
                cm = cm / cm.sum(axis=1, keepdims=True)
            elif normalize == "pred":
                cm = cm / cm.sum(axis=0, keepdims=True)
            elif normalize == "all":
                cm = cm / cm.sum()
        cm = np.nan_to_num(cm)

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
        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight")
        buf.seek(0)
        return PIL.Image.open(buf)

    def log(self, logger_object) -> None:
        if isinstance(logger_object, pl_loggers.WandbLogger):
            # writing to buffer is necessary to avoid wandb cutting our labels off
            fig = self.plot()
            image = self._fig2img(fig)
            logger_object.experiment.log({"Confusion Matrix": wandb.Image(image)})
        self._reset()
        return
