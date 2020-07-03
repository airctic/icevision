__all__ = ["convert_records_to_coco_style", "coco_api_from_records"]

from pycocotools.coco import COCO
from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.core import *


def coco_api_from_records(records):
    """ Create pycocotools COCO dataset from records
    """
    coco_ds = COCO()
    coco_ds.dataset = convert_records_to_coco_style(records)
    coco_ds.createIndex()
    return coco_ds


def convert_records_to_coco_style(records):
    """ Converts records from library format to coco format.
    Inspired from: https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py#L146
    """
    images = []
    annotations_dict = defaultdict(list)
    categories_set = set()

    for record in pbar(records):
        # build images field
        image = {}
        image["id"] = record["imageid"]
        image["file_name"] = Path(record["filepath"]).name
        image["width"] = record["width"]
        image["height"] = record["height"]
        images.append(image)

        # build annotations field
        for label in record["labels"]:
            annotations_dict["image_id"].append(record["imageid"])
            annotations_dict["category_id"].append(label)
            categories_set.add(label)

        if "bboxes" in record:
            for bbox in record["bboxes"]:
                annotations_dict["bbox"].append(bbox.xywh)

        if "areas" in record:
            for area in record["areas"]:
                annotations_dict["area"].append(area)
        else:
            for bbox in record["bboxes"]:
                annotations_dict["area"].append(bbox.areas)

        if "masks" in record:
            for mask in record["masks"]:
                if isinstance(mask, Polygon):
                    annotations_dict["segmentation"].append(mask.points)
                elif isinstance(mask, RLE):
                    coco_rle = {
                        "counts": mask.to_coco(),
                        "size": [record["height"], record["width"]],
                    }
                    annotations_dict["segmentation"].append(coco_rle)
                elif isinstance(mask, MaskFile):
                    rles = mask.to_coco_rle(record["height"], record["width"])
                    annotations_dict["segmentation"].extend(rles)
                else:
                    msg = f"Mask type {type(mask)} unsupported, we only support RLE and Polygon"
                    raise ValueError(msg)

        # TODO: is auto assigning a value for iscrowds dangerous (may hurt the metric value?)
        if "iscrowds" not in record:
            record["iscrowds"] = [0] * len(record["labels"])
        for iscrowd in record["iscrowds"]:
            annotations_dict["iscrowd"].append(iscrowd)

    if not allequal([len(o) for o in annotations_dict.values()]):
        raise RuntimeError("Mismatch lenght of elements")

    categories = [{"id": i} for i in categories_set]

    # convert dict of lists to list of dicts
    annotations = []
    for i in range(len(annotations_dict["image_id"])):
        annotation = {k: v[i] for k, v in annotations_dict.items()}
        # annotations should be initialized starting at 1 (torchvision issue #1530)
        annotation["id"] = i + 1
        annotations.append(annotation)

    return {"images": images, "annotations": annotations, "categories": categories}
