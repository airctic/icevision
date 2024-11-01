__all__ = ["Task", "common", "detection", "classification"]

from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class Task:
    name: str
    order: float = 0.5


common = Task("common", 0.3)
detection = Task("detection")
classification = Task("classification")
segmentation = Task("segmentation")
