__all__ = ["InvalidDataError", "AutofixAbort", "AbortParseRecord"]


class InvalidDataError(Exception):
    pass


class AutofixAbort(Exception):
    pass


class AbortParseRecord(Exception):
    pass
