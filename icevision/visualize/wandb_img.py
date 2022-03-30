__all__ = ["wandb_img_preds", "wandb_image"]


from typing import List

import wandb
from icevision import BaseRecord, BBox
from icevision.data.prediction import Prediction


def wandb_img_preds(
    preds: List[Prediction], add_ground_truth: bool = False
) -> List[wandb.Image]:
    return [wandb_image(pred, add_ground_truth=add_ground_truth) for pred in preds]


def bbox_wandb(bbox: BBox, label_id: int, label_name: str, score=None) -> dict:
    """Return a wandb compatible dictionary with bbox, label and score"""
    xmin, ymin, xmax, ymax = map(int, bbox.xyxy)

    box_data = {
        "position": {"minX": xmin, "maxX": xmax, "minY": ymin, "maxY": ymax},
        "class_id": int(label_id),
        "domain": "pixel",
    }

    if score:
        score = int(score * 100)
        box_caption = f"{label_name} ({score}%)"
        box_data["score"] = score
    else:
        box_caption = label_name

    box_data["box_caption"] = box_caption

    return box_data


def wandb_image(pred: Prediction, add_ground_truth: bool = False) -> wandb.Image:
    """Return a wandb image corresponding to the a prediction.

    Args:
        pred (Prediction): A prediction to log with WandB.
            Must have been created with keep_image = True.
        add_ground_truth (bool, optional): Add ground_truth information to the
            the WandB image. Defaults to False.

    Returns:
        wandb.Image: Specifying the image, but also the predictions and  possibly ground_truth.
    """
    # FIXME: if pred does not have an img, then we lose.
    # FIXME: Not handling masks

    # Check if "masks" key is the sample dictionnary
    # if "masks" in sample:     true_masks = sample["masks"]

    # Check if "masks" key is the pred dictionnary
    # if "masks" in pred: pred_masks = pred["masks"]

    class_id_to_label = {
        id: label for id, label in enumerate(pred.detection.class_map._id2class)
    }

    # Prediction
    box_data = list(
        map(
            bbox_wandb,
            pred.detection.bboxes,
            pred.detection.label_ids,
            pred.detection.labels,
            pred.detection.scores,
        )
    )

    boxes = {"predictions": {"box_data": box_data, "class_labels": class_id_to_label}}

    # Predicted Masks
    # Check if "masks" key is the pred dictionnary
    #     if "masks" in pred:
    #         mask_data = (pred_masks.data * pred["labels"][:, None, None]).max(0)
    #         masks = {
    #             "predictions": {"mask_data": mask_data, "class_labels": class_id_to_label}
    #         }
    #     else:
    #         masks = None
    masks = None

    # Ground Truth
    if add_ground_truth:
        box_data = list(
            map(
                bbox_wandb,
                pred.ground_truth.detection.bboxes,
                pred.ground_truth.detection.label_ids,
                pred.ground_truth.detection.labels,
            )
        )

        boxes["ground_truth"] = {
            "box_data": box_data,
            "class_labels": class_id_to_label,
        }

        # # Ground Truth Masks
        # Check if "masks" key is the sample dictionnary
        #         if "masks" in sample:
        #             labels_arr = np.array(sample["labels"])
        #             mask_data = (true_masks.data * labels_arr[:, None, None]).max(0)
        #             masks["ground_truth"] = {
        #                 "mask_data": mask_data,
        #                 "class_labels": class_id_to_label,
        #             }
    return wandb.Image(pred.img, boxes=boxes, masks=masks)
