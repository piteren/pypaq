import logging
from typing import Optional

from pypaq.lipytools.little_methods import stamp, prep_folder


# returns formatted Logger
def get_pylogger(
        name: str,
        add_stamp=                  True,
        folder: Optional[str]=      None,   # if given then writes logfile
        level=                      logging.INFO,
        format=                     '%(asctime)s {%(filename)s:%(lineno)d} p%(process)s %(levelname)s: %(message)s',
        to_stdout=                  True):

    # add TRACE (9) level
    if "TRACE" not in logging.__all__:
        logging.addLevelName(9, "TRACE")
        logging.__all__.append("TRACE")
        def trace(self, message, *args, **kws):
            if self.isEnabledFor(9):
                self._log(9, message, args, **kws)
        logging.Logger.trace = trace

    if add_stamp: name += '_' + stamp()

    formatter = logging.Formatter(format)

    # manage file handler
    fh = None
    if folder:
        prep_folder(folder)
        fh = logging.FileHandler(f'{folder}/{name}.log')
        fh.setFormatter(formatter)

    # manage stream handler
    sh = None
    if to_stdout:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if fh: logger.addHandler(fh)
    if sh: logger.addHandler(sh)
    return logger

# returns higher level of logger
def get_higher(level: int) -> int:
    if level == 9: return 10
    level += 10
    if level > 50: level = 50
    return level

# returns child with higher level
def get_hi_child(logger, name):
    clogger = logger.getChild(name)
    lvl = clogger.getEffectiveLevel()
    clogger.setLevel(get_higher(lvl))
    return clogger