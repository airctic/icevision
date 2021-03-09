__all__ = ["Task", "default", "detect", "classif"]

from icevision.imports import *


@dataclass(eq=True, frozen=True)
class Task:
    name: str
    order: float = 0.5


default = Task("default", 0.3)
detect = Task("detect")
classif = Task("classif")
