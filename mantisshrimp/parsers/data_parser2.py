from mantisshrimp import *
from mantisshrimp.imports import *
from mantisshrimp.imports import ABC


import mantisshrimp


def parse(image_info_parser, annotation_parser, data_splitter):
    # parse image infos
    infos = image_info_parser.parse()
    infos_ids = set(infos.keys())
    # parse annotations
    annotations = annotation_parser.parse()
    annotations_ids = set(annotations.keys())
    # removes ids that are not included in both
    valid_ids = infos_ids.intersection(annotations_ids)
    excluded = infos_ids.union(annotations_ids) - valid_ids
    print(f"Removed {excluded}")
    # combine image_info with annotations and separate splits
    splits = data_splitter(valid_ids, [0.9, 0.1])
    return [
        {id: {"image_info": infos[id], "annotation": annotations[id]} for id in ids}
        for ids in splits
    ]


source = Path(mantisshrimp.__file__).parent.parent / "samples"
annots_dict = json.loads((source / "annotations.json").read())

image_info_parser = COCOImageInfoParser(annots_dict["images"], source)
coco_annotation_parser = COCOAnnotationParser2(annots_dict["annotations"])

train_rs, valid_rs = parse(image_info_parser, coco_annotation_parser, random_split2)

len(train_rs), len(valid_rs)
