import functools
import logging

from pypaq.lipytools.files import prep_folder

DEFAULT_FMT = '%(asctime)s {%(filename)20s:%(lineno)4d} p%(process)s %(levelname)s: %(message)s'


def logger_mod(
        level: int | None = logging.INFO,
            # handlers
        folder: str | None = None,
        log_file_name: str | None = None,
        to_stdout: bool = True,
        replace_stream: bool = False,
            # format
        fmt: str = DEFAULT_FMT,
        file_width: int = 20,
        enable_process: bool = True,
) -> None:
    """modify the root logger level, handlers and format
    call once at app startup

    replace_stream: if True, removes any existing console StreamHandler(s)
        first, so this call's StreamHandler replaces them (instead of being
        skipped by the has_stream guard)"""

    logger = logging.getLogger()

    if level is not None:
        logger.setLevel(level)

    if "(filename)20s" in fmt:
        fmt = fmt.replace("(filename)20s",f"(filename){file_width}s")
    if not enable_process and "p%(process)s " in fmt:
        fmt = fmt.replace("p%(process)s ", "")
    formatter = logging.Formatter(fmt)

    if to_stdout:

        if replace_stream:
            for h in [h for h in logger.handlers if type(h) is logging.StreamHandler]:
                logger.removeHandler(h)

        has_stream = any(type(h) is logging.StreamHandler for h in logger.handlers)
        if not has_stream:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger.addHandler(sh)

    if folder:
        prep_folder(folder)
        log_file_name = log_file_name or logger.name
        if not log_file_name.endswith(".log"):
            log_file_name += ".log"
        fh = logging.FileHandler(f'{folder}/{log_file_name}')
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def shift_log_level(logger, delta):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            old = logger.getEffectiveLevel()
            logger.setLevel(old + delta)
            try:
                return fn(*args, **kwargs)
            finally:
                logger.setLevel(old)
        return wrapper
    return decorator
