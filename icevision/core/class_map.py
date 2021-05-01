__all__ = ["ClassMap", "BACKGROUND"]

from icevision.imports import *

BACKGROUND = "background"


class ClassMap:
    """Utility class for mapping between class name and id."""

    def __init__(
        self,
        classes: Optional[Sequence[str]] = None,
        background: Optional[str] = BACKGROUND,
    ):
        self._lock = True

        self._id2class = copy(list(classes)) if classes else []
        # insert background if required
        self._background = background
        if self._background is not None:
            try:
                self._id2class.remove(self._background)
            except ValueError:
                pass
            # background is always index zero
            self._id2class.insert(0, self._background)

        self._class2id = {name: i for i, name in enumerate(self._id2class)}

    @property
    def num_classes(self):
        return len(self)

    def get_by_id(self, id: int) -> str:
        return self._id2class[id]

    def get_by_name(self, name: str) -> int:
        try:
            return self._class2id[name]
        except KeyError as e:
            if not self._lock:
                return self.add_name(name)
            else:
                raise e

    def add_name(self, name: str) -> int:
        # Raise error if trying to add duplicate value
        if name in self._id2class:
            raise ValueError(
                f"'{name}' already exists in the ClassMap. You can only add new labels that are unique"
            )

        self._id2class.append(name)
        id = len(self._class2id)
        self._class2id[name] = id
        return id

    def lock(self):
        self._lock = True
        return self

    def unlock(self):
        self._lock = False
        return self

    def __eq__(self, other) -> bool:
        if isinstance(other, ClassMap):
            return self.__dict__ == other.__dict__
        return False

    def __len__(self):
        return len(self._id2class)

    def __repr__(self):
        return f"<ClassMap: {self._class2id.__repr__()}>"
