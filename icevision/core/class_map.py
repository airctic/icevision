__all__ = ["ClassMap"]

from icevision.imports import *


class ClassMap:
    """Utility class for mapping between class name and id."""

    def __init__(self, classes: List[str], background: Optional[int] = 0):
        classes = classes.copy()

        if background is not None:
            if background == -1:
                background = len(classes)
            classes.insert(background, "background")

        self.id2class = classes
        self.class2id = {name: i for i, name in enumerate(classes)}

    def __len__(self):
        return len(self.id2class)

    def get_id(self, id: int) -> str:
        return self.id2class[id]

    def get_name(self, name: str) -> int:
        return self.class2id[name]

    def __repr__(self):
        return f"<ClassMap: {self.class2id.__repr__()}>"
