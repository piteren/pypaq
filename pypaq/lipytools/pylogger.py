import logging
from typing import Optional

from pypaq.lipytools.little_methods import stamp, prep_folder


# returns formatted Logger
def get_pylogger(
        name: str,
        add_stamp=              True,
        folder: Optional[str]=  None,   # if given then writes logfile
        level=                  logging.INFO,
        format: str=            '%(asctime)s {%(filename)17s:%(lineno)3d} p%(process)s %(levelname)s: %(message)s',
        to_stdout=              True):

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

# returns child with higher level
def get_hi_child(logger, name, higher_level=True):
    clogger = logger.getChild(name)
    if higher_level:
        lvl = clogger.getEffectiveLevel()
        clogger.setLevel(lvl+10)
    return clogger