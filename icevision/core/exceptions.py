__all__ = [
    "InvalidDataError",
    "AutofixAbort",
    "AbortParseRecord",
    "InvalidMMSegModelType",
]


class InvalidDataError(Exception):
    pass


class AutofixAbort(Exception):
    pass


class AbortParseRecord(Exception):
    pass


class InvalidMMSegModelType(Exception):
    pass