__all__ = ["CombinedParser"]


class CombinedParser:
    def __init__(self, image_info_parser, annotation_parser):
        self.image_info_parser = image_info_parser
        self.annotation_parser = annotation_parser

    def parse(self, data_splitter, show_pbar: bool = True):
        infos = self.image_info_parser.parse(show_pbar=show_pbar)
        annotations = self.annotation_parser.parse(show_pbar=show_pbar)
        infos_ids = set(infos.keys())
        annotations_ids = set(annotations.keys())
        # removes ids that are not included in both
        valid_ids = infos_ids.intersection(annotations_ids)
        excluded = infos_ids.union(annotations_ids) - valid_ids
        print(f"Removed {excluded}")
        # combine image_info with annotations and separate splits
        splits = data_splitter(valid_ids)
        return [
            [{"imageid": id, **infos[id], **annotations[id]} for id in ids]
            for ids in splits
        ]
