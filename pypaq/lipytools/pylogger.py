import logging

from pypaq.lipytools.files import prep_folder

DEFAULT_FMT = '%(asctime)s {%(filename)20s:%(lineno)4d} p%(process)s %(levelname)s: %(message)s'


def logging_mod(
        level: int | None = None,
        folder: str | None = None,
        log_file_name: str | None = None,
        to_stdout: bool = True,
        fmt: str = DEFAULT_FMT,
) -> None:
    """Configure the root logger handlers.
    Call once at app startup."""

    logger = logging.getLogger()

    if level is not None:
        logger.setLevel(level)

    formatter = logging.Formatter(fmt)

    has_stream = any(type(h) is logging.StreamHandler for h in logger.handlers)
    if to_stdout and not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if folder:
        prep_folder(folder)
        fh = logging.FileHandler(f'{folder}/{log_file_name or logger.name}.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)