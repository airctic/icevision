__all__ = ["logger_default_config", "autofix_log", "ReplaySink"]

from icevision.imports import *

logger.level("AUTOFIX", 25)
logger.level("AUTOFIX-START", 25, color="<fg #1f6f8b><bold>")
logger.level("AUTOFIX-SUCCESS", 25, color="<green>")
logger.level("AUTOFIX-FAIL", 25, color="<fg #ea2c62>")
logger.level("AUTOFIX-REPORT", 25, color="<fg #9a7d0a>")


def autofix_log(
    level: str, message: str, *args, record_id: Optional[Any] = None
) -> None:
    if record_id is not None:
        message = f"<b><red>(record_id: {record_id})</></> - {message}"
    logger.opt(colors=True).log(level, message, *args)


def logger_default_config():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level><bold>{level: <8}</></> - <level>{message}</> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>",
        level="INFO",
        colorize=True,
    )


logger_default_config()


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
        def _noop():
            pass

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
        logger_default_config()
        if self.captured:
            self.pre_replay()

            for msg in self.captured:
                record = msg.record

                level = record["level"].name
                try:
                    logger.level(level)
                except ValueError:
                    # If the level 'name' is not registered, use its 'no' instead
                    level = record["level"].no

                patched = logger.patch(lambda r: r.update(record))
                patched.log(level, record["message"])

            self.post_replay()
