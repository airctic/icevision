__all__ = [
    "InvalidDataError",
    "AutofixAbort",
    "AbortParseRecord",
    "InvalidMMSegModelType",
    "PreTrainedVariantNotFound",
]


class InvalidDataError(Exception):
    pass


class AutofixAbort(Exception):
    pass


class AbortParseRecord(Exception):
    pass


class InvalidMMSegModelType(Exception):
    pass


class PreTrainedVariantNotFound(Exception):
    pass
