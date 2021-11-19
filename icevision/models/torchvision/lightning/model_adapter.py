__all__ = ["RCNNModelAdapter"]

from icevision.imports import *
from icevision.utils import *
from icevision.metrics import *
from icevision.engines.lightning.lightning_model_adapter import LightningModelAdapter
from icevision.models.torchvision.loss_fn import loss_fn
from icevision.core.record_components import (
    InstanceMasksRecordComponent,
    BBoxesRecordComponent,
)
from icevision.core.mask import MaskArray
from icevision.core.bbox import BBox


class RCNNModelAdapter(LightningModelAdapter, ABC):
    def __init__(self, model: nn.Module, metrics: Sequence[Metric] = None):
        super().__init__(metrics=metrics)
        self.model = model

    @abstractmethod
    def convert_raw_predictions(self, batch, raw_preds, records):
        """Convert raw predictions from the model to library standard."""

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        (xb, yb), records = batch
        preds = self(xb, yb)

        loss = loss_fn(preds, yb)
        self.log("train_loss", loss)

        return loss

    @staticmethod
    def get_component_by_class(components, component_class):
        for component in components:
            if isinstance(component, component_class):
                return component
        else:
            return None

    def update_records(self, records, targets):
        new_records = []
        for record, target in zip(records, targets):
            record_copy = deepcopy(record)
            instance_mask_record_component = self.get_component_by_class(
                record_copy.detection.components, InstanceMasksRecordComponent
            )
            if instance_mask_record_component is not None:
                mask_array = MaskArray(target["masks"].cpu().numpy())
                instance_mask_record_component.set_mask_array(mask_array)
                instance_mask_record_component.set_masks(
                    [mask_array.to_erles(None, None)]
                )
            bbox_record_component = self.get_component_by_class(
                record_copy.detection.components, BBoxesRecordComponent
            )
            if bbox_record_component is not None:
                bbox_record_component.set_bboxes(
                    [BBox(*row.cpu().numpy().tolist()) for row in target["boxes"]]
                )
            new_records.append(record_copy)
        return new_records

    def validation_step(self, batch, batch_idx):
        (xb, yb), records = batch
        updated_records = self.update_records(records, yb)
        with torch.no_grad():
            self.train()
            train_preds = self(xb, yb)
            loss = loss_fn(train_preds, yb)

            self.eval()
            raw_preds = self(xb)
            preds = self.convert_raw_predictions(
                batch=batch, raw_preds=raw_preds, records=updated_records
            )
            self.accumulate_metrics(preds=preds)

        self.log("val_loss", loss)

    def validation_epoch_end(self, outs):
        self.finalize_metrics()
