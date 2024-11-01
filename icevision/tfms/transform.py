__all__ = ["Transform"]

from abc import ABC, abstractmethod

from icevision.core.record import BaseRecord


class Transform(ABC):
    def __call__(self, record: BaseRecord):
        # TODO: this assumes record is already loaded and copied
        # which is generally true
        return self.apply(record)

    @abstractmethod
    def apply(self, record: BaseRecord) -> BaseRecord:
        """Apply the transform
        Returns:
              dict: Modified values, the keys of the dictionary should have the same
              names as the keys received by this function
        """
