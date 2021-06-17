__all__ = ["predict", "predict_from_dl", "convert_raw_predictions", "end2end_detect"]

from icevision.imports import *
from icevision.utils import *
from icevision.core import *
from icevision.data import *
from icevision.models.utils import _predict_from_dl
from icevision.models.ross.efficientdet.dataloaders import *
from effdet import DetBenchTrain, DetBenchPredict, unwrap_bench
from icevision.models.inference import *


@torch.no_grad()
def _predict_batch(
    model: Union[DetBenchTrain, DetBenchPredict],
    batch: Sequence[torch.Tensor],
    records: Sequence[BaseRecord],
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:
    device = device or model_device(model)

    imgs, img_info = batch
    imgs = imgs.to(device)
    img_info = {k: v.to(device) for k, v in img_info.items()}

    bench = DetBenchPredict(unwrap_bench(model))
    bench = bench.eval().to(device)

    raw_preds = bench(x=imgs, img_info=img_info)
    preds = convert_raw_predictions(
        batch=batch,
        raw_preds=raw_preds,
        records=records,
        detection_threshold=detection_threshold,
        keep_images=keep_images,
    )

    return preds


def predict(
    model: Union[DetBenchTrain, DetBenchPredict],
    dataset: Dataset,
    detection_threshold: float = 0.5,
    keep_images: bool = False,
    device: Optional[torch.device] = None,
) -> List[Prediction]:

    batch, records = build_infer_batch(dataset)
    return _predict_batch(
        model=model,
        batch=batch,
        records=records,
        detection_threshold=detection_threshold,
        keep_images=keep_images,
        device=device,
    )


def predict_from_dl(
    model: nn.Module,
    infer_dl: DataLoader,
    show_pbar: bool = True,
    keep_images: bool = False,
    **predict_kwargs,
):
    return _predict_from_dl(
        predict_fn=_predict_batch,
        model=model,
        infer_dl=infer_dl,
        show_pbar=show_pbar,
        keep_images=keep_images,
        **predict_kwargs,
    )


def convert_raw_predictions(
    batch,
    raw_preds: torch.Tensor,
    records: Sequence[BaseRecord],
    detection_threshold: float,
    keep_images: bool = False,
) -> List[Prediction]:
    tensor_images, *_ = batch
    dets = raw_preds.detach().cpu().numpy()
    preds = []
    for det, record, tensor_image in zip(dets, records, tensor_images):
        if detection_threshold > 0:
            scores = det[:, 4]
            keep = scores > detection_threshold
            det = det[keep]

        pred = BaseRecord(
            (
                ScoresRecordComponent(),
                ImageRecordComponent(),
                InstancesLabelsRecordComponent(),
                BBoxesRecordComponent(),
            )
        )

        pred.detection.set_class_map(record.detection.class_map)
        pred.detection.set_labels_by_id(det[:, 5].astype(int))
        pred.detection.set_bboxes([BBox.from_xyxy(*xyxy) for xyxy in det[:, :4]])
        pred.detection.set_scores(det[:, 4])

        if keep_images:
            record.set_img(tensor_to_image(tensor_image))

        preds.append(Prediction(pred=pred, ground_truth=record))

    return preds


end2end_detect = partial(_end2end_detect, predict_fn=predict)
