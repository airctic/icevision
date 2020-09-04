__all__ = ["CaptureStdout"]

from icevision.imports import *
from io import StringIO


class CaptureStdout(list):
    """Capture the stdout (like prints)
    From: https://stackoverflow.com/a/16571630/6772672
    """

    def __init__(self, propagate_stdout: bool = False):
        self.propagate_stdout = propagate_stdout

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout

        if self.propagate_stdout:
            print("\n".join(self))
