__all__ = ["MMDetModelAdapter"]

from icevision.imports import *
from icevision.utils import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.core.record_components import (
    InstanceMasksRecordComponent,
    BBoxesRecordComponent,
)
from icevision.core.mask import MaskArray
from icevision.core.bbox import BBox

from icevision.models.mmdet.common.bbox import convert_raw_predictions


class MMDetModelAdapter(LightningModelAdapter, ABC):
    """Lightning module specialized for EfficientDet, with metrics support.

    The methods `forward`, `training_step`, `validation_step`, `validation_epoch_end`
    are already overriden.

    # Arguments
        model: The pytorch model to use.
        metrics: `Sequence` of metrics to use.

    # Returns
        A `LightningModule`.
    """

    def __init__(self, model: nn.Module, metrics: List[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model

    @abstractmethod
    def convert_raw_predictions(self, batch, raw_preds, records):
        """Convert raw predictions from the model to library standard."""

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        data, samples = batch

        outputs = self.model.train_step(data=data, optimizer=None)

        for k, v in outputs["log_vars"].items():
            self.log(f"train/{k}", v)

        return outputs["loss"]

    @staticmethod
    def get_component_by_class(components, component_class):
        for component in components:
            if isinstance(component, component_class):
                return component
        else:
            return None

    def get_updated_records(self, records, masks):
        new_records = []
        for record, mask in zip(records, masks):
            record_copy = deepcopy(record)
            instance_mask_record_component = self.get_component_by_class(
                record_copy.detection.components, InstanceMasksRecordComponent
            )
            if instance_mask_record_component is not None:
                mask_array = MaskArray(mask.masks)
                instance_mask_record_component.set_mask_array(mask_array)
                instance_mask_record_component.set_masks(
                    mask_array.to_erles(None, None).erles
                )
            bbox_record_component = self.get_component_by_class(
                record_copy.detection.components, BBoxesRecordComponent
            )
            if bbox_record_component is not None:
                bbox_record_component.set_bboxes(
                    [BBox(*row.tolist()) for row in mask.get_bboxes()]
                )
            new_records.append(record_copy)
        return new_records

    def validation_step(self, batch, batch_idx):
        data, records = batch

        if "gt_masks" in data.keys():
            updated_records = self.get_updated_records(records, data["gt_masks"])
        else:
            updated_records = records

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.train_step(data=data, optimizer=None)
            raw_preds = self.model.forward_test(
                imgs=[data["img"]], img_metas=[data["img_metas"]]
            )

        preds = self.convert_raw_predictions(
            batch=data, raw_preds=raw_preds, records=updated_records
        )
        self.accumulate_metrics(preds)

        for k, v in outputs["log_vars"].items():
            self.log(f"valid/{k}", v)

        # TODO: is train and eval model automatically set by lighnting?
        self.model.train()

    def validation_epoch_end(self, outs):
        self.finalize_metrics()
