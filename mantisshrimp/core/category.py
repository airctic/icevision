from ..imports import *
from ..utils import *

__all__ = ["Category", "CategoryMap"]


@dataclass(frozen=True)
class Category:
    id: int = None
    name: str = None


class CategoryMap:
    def __init__(self, cats, background=None):
        self.cats = cats.copy()
        if notnone(background):
            self.cats.pop(background)
        self.cats.insert(0, background or Category(name="background"))
        self.i2o = {i: o for i, o in enumerate(self.cats)}
        self.o2i = {o: i for i, o in enumerate(self.cats)}
        self.id2i = {o.id: i for i, o in enumerate(self.cats) if notnone(o.id)}
        self.id2o = {o.id: o for o in self.cats if notnone(o.id)}

    def __len__(self):
        return len(self.cats)

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self)} categories>"
