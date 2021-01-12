__all__ = ["ClassMap", "BACKGROUND"]

from icevision.imports import *

BACKGROUND = "background"


class ClassMap:
    """Utility class for mapping between class name and id."""

    def __init__(self, classes: Optional[Sequence[str]] = None):
        self._classes = copy(classes) if classes else []
        self._lock = True
        self._background_id = None
        self._update_ids()

    def _update_ids(self):
        self._id2class = copy(self._classes)
        if self._background_id is not None:
            background_id = (
                self._background_id if self._background_id != -1 else len(self._classes)
            )
            self._id2class.insert(background_id, BACKGROUND)

        self._class2id = {name: i for i, name in enumerate(self._id2class)}

    def get_id(self, id: int) -> str:
        return self._id2class[id]

    def get_name(self, name: str) -> int:
        try:
            return self._class2id[name]
        except KeyError as e:
            if not self._lock:
                self._classes.append(name)
                self._update_ids()
                return self._class2id[name]
            else:
                raise e

    def set_background(self, id: Union[int, None]):
        self._background_id = id
        self._update_ids()

    def lock(self):
        self._lock = True

    def unlock(self):
        self._lock = False

    def __len__(self):
        return len(self._id2class)

    def __repr__(self):
        return f"<ClassMap: {self._class2id.__repr__()}>"
