__all__ = ["ReplaySink"]

from icevision.imports import *


def _noop():
    pass


class ReplaySink:
    """Capture messages and replays them after leaving the block.

    # Examples
    ```python
    pre_replay = lambda: logger.info("something will happen")
    post_replay = lambda: logger.info("something did happen")
    with ReplaySink(pre_replay=pre_replay, post_replay=post_replay) as sink:
        logger.info('captured message')
    ```
    """

    def __init__(
        self,
        pre_replay: Optional[callable] = None,
        post_replay: Optional[callable] = None,
    ):
        self.captured = []
        self.pre_replay = pre_replay or _noop
        self.post_replay = post_replay or _noop

    def __enter__(self):
        logger.remove()
        logger.add(self)
        return self

    def write(self, message):
        self.captured.append(message)

    def __exit__(self, type, value, traceback):
        logger.remove()
        logger.add(sys.stderr)
        if self.captured:
            self.pre_replay()
            for msg in self.captured:
                record = msg.record
                patched = logger.patch(lambda r: r.update(record))
                patched.log(record["level"].no, record["message"])
            self.post_replay()
