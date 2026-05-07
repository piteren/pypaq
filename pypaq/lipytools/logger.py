import logging

from pypaq.lipytools.files import prep_folder

DEFAULT_FMT = '%(asctime)s {%(filename)20s:%(lineno)4d} p%(process)s %(levelname)s: %(message)s'


def logger_mod(
        level: int | None = logging.INFO,
            # handlers
        folder: str | None = None,
        log_file_name: str | None = None,
        to_stdout: bool = True,
            # format
        fmt: str = DEFAULT_FMT,
        file_width: int = 20,
        enable_process: bool = True,
) -> None:
    """modify the root logger level, handlers and format
    call once at app startup"""

    logger = logging.getLogger()

    if level is not None:
        logger.setLevel(level)

    if "(filename)20s" in fmt:
        fmt = fmt.replace("(filename)20s",f"(filename){file_width}s")
    if not enable_process and "p%(process)s " in fmt:
        fmt = fmt.replace("p%(process)s ", "")
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