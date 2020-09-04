__all__ = ["IDMap"]

from icevision.imports import *


class IDMap:
    """
    Works like a dictionary that automatically assign values for new keys.
    """

    def __init__(self, initial_names: Optional[Sequence[Hashable]] = None):
        names = initial_names or []
        self.id2name = OrderedDict((id, name) for id, name in enumerate(names))
        self.name2id = OrderedDict((name, id) for id, name in enumerate(names))

    def get_id(self, id: int) -> Hashable:
        return self.id2name[id]

    def get_name(self, name: Hashable) -> int:
        try:
            id = self.name2id[name]
        except KeyError:
            id = len(self.name2id)
            self.name2id[name] = id
            self.id2name[id] = name

        return id

    def filter_ids(self, ids: List[int]) -> "IDMap":
        idmap = IDMap()
        for id in ids:
            name = self.get_id(id)
            idmap.id2name[id] = name
            idmap.name2id[name] = id

        return idmap

    def get_ids(self) -> List[int]:
        return list(self.id2name.keys())

    def get_names(self) -> List[Hashable]:
        return list(self.name2id.keys())

    def __getitem__(self, imageid):
        return self.get_name(imageid)
