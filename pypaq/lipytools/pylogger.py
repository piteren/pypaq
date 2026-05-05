import logging

from pypaq.lipytools.files import prep_folder

DEFAULT_FMT = '%(asctime)s {%(filename)20s:%(lineno)4d} p%(process)s %(levelname)s: %(message)s'


def setup_logging(
        level: int = logging.INFO,
        folder: str | None = None,
        log_file_name: str | None = None,
        to_stdout: bool = True,
        fmt: str = DEFAULT_FMT,
) -> None:
    """Configure the root logger handlers.
    Call once at app startup."""

    logger = logging.getLogger()
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


class Logged:
    """Manages class/object logger.
    For classes with nuanced logging levels or folder logging."""

    def get_logger(
            self,
            level: int | None = None,
            folder: str | None = None,
            fmt: str = DEFAULT_FMT,
    ) -> logging.Logger:
        """Call in class init just after setting self.name,
        if setting self.name is not set, logger becomes class (not object) logger."""

        logger_name = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
        obj_name = getattr(self, 'name', None)
        if obj_name:
            logger_name += f'.{obj_name}'
        logger = logging.getLogger(logger_name)

        if level is not None:
            logger.setLevel(level)

        # prevents duplicated FileHandlers
        has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        if folder and not has_file:
            prep_folder(folder)
            formatter = logging.Formatter(fmt)
            fh = logging.FileHandler(f'{folder}/{obj_name or self.__class__.__qualname__}.log')
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger