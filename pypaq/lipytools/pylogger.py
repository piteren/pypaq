import logging
from typing import Optional

from pypaq.lipytools.little_methods import stamp, prep_folder


# returns formatted Logger
def get_pylogger(
        name: str,
        add_stamp=                  True,
        folder: Optional[str]=      None,   # if given then writes logfile
        level=                      logging.INFO,
        format: Optional[str]=      None,
        to_stdout=                  True):

    if add_stamp: name += '_' + stamp()

    if not format:
        format = '%(asctime)s %(levelname)s: %(message)s'
        if level < 20: format = '%(asctime)s {%(filename)s:%(lineno)d} p%(process)s %(levelname)s: %(message)s'
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
def get_hi_child(logger, name):
    clogger = logger.getChild(name)
    lvl = clogger.getEffectiveLevel()
    clogger.setLevel(lvl+10)
    return clogger