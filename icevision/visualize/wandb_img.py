__all__ = [
    "wandb_img_preds",
]

from icevision.imports import *
from icevision.data import *
from icevision.core import *


def bbox_wandb(bbox: BBox, label: int, class_id_to_label, pred_score=None):
    """Creates a wandb compatible dictionary with bbox, label and score"""
    xmin, ymin, xmax, ymax = map(int, bbox.xyxy)

    box_data = {
        "position": {"minX": xmin, "maxX": xmax, "minY": ymin, "maxY": ymax},
        "class_id": label,
        "domain": "pixel",
    }

    if pred_score:
        score = int(pred_score * 100)
        box_caption = f"{class_id_to_label[label]} ({score}%)"
        box_data["score"] = score
    else:
        box_caption = f"{class_id_to_label[label]}"

    box_data["box_caption"] = box_caption

    return box_data


def wandb_image(sample, pred, class_id_to_label, add_ground_truth=False):
    raw_image = sample["img"]
    true_bboxes = sample["bboxes"]
    true_labels = sample["labels"]
    # Check if "masks" key is the sample dictionnary
    if "masks" in sample:
        true_masks = sample["masks"]

    pred_bboxes = pred["bboxes"]
    pred_labels = pred["labels"].tolist()
    pred_scores = pred["scores"]
    # Check if "masks" key is the pred dictionnary
    if "masks" in pred:
        pred_masks = pred["masks"]

    # Predicted Boxes
    pred_all_boxes = []
    # Collect predicted bounding boxes for this image
    for b_i, bbox in enumerate(pred_bboxes):
        box_data = bbox_wandb(
            bbox, pred_labels[b_i], class_id_to_label, pred_score=pred_scores[b_i]
        )
        pred_all_boxes.append(box_data)

    # log to wandb: raw image, predictions, and dictionary of class labels for each class id
    boxes = {
        "predictions": {"box_data": pred_all_boxes, "class_labels": class_id_to_label}
    }

    # Predicted Masks
    # Check if "masks" key is the pred dictionnary
    if "masks" in pred:
        mask_data = (pred_masks.data * pred["labels"][:, None, None]).max(0)
        masks = {
            "predictions": {"mask_data": mask_data, "class_labels": class_id_to_label}
        }
    else:
        masks = None

    # Ground Truth
    if add_ground_truth:
        # Ground Truth Boxes
        true_all_boxes = []
        # Collect ground truth bounding boxes for this image
        for b_i, bbox in enumerate(true_bboxes):
            box_data = bbox_wandb(bbox, true_labels[b_i], class_id_to_label)
            true_all_boxes.append(box_data)

        boxes["ground_truth"] = {
            "box_data": true_all_boxes,
            "class_labels": class_id_to_label,
        }

        # # Ground Truth Masks
        # Check if "masks" key is the sample dictionnary
        if "masks" in sample:
            labels_arr = np.array(sample["labels"])
            mask_data = (true_masks.data * labels_arr[:, None, None]).max(0)
            masks["ground_truth"] = {
                "mask_data": mask_data,
                "class_labels": class_id_to_label,
            }

    return wandb.Image(raw_image, boxes=boxes, masks=masks)


def wandb_img_preds(samples, preds, class_map, add_ground_truth=False):
    class_id_to_label = {int(v): k for k, v in class_map.class2id.items()}

    wandb_imgs = []
    for (sample, pred) in zip(samples, preds):
        img_wandb = wandb_image(
            sample, pred, class_id_to_label, add_ground_truth=add_ground_truth
        )
        wandb_imgs.append(img_wandb)
    return wandb_imgs
