__all__ = ["BatchTransform"]


class BatchTransform(ABC):
    def __call__(self, records: List[RecordType]) -> List[RecordType]:
        return self.apply(records=records)

    @abstractmethod
    def apply(self, records: List[RecordType]) -> List[RecordType]:
        """Main transform logic.

        Args:
            records (List[RecordType]): List of records to apply transforms.
        """
