import logging
from typing import Optional, Union, Tuple

from pypaq.lipytools.printout import stamp
from pypaq.lipytools.files import prep_folder


# returns formatted Logger
def get_pylogger(
        name: Optional[str]=                None,
        add_stamp=                          True,
        folder: Optional[str]=              None,
        level=                              logging.INFO,
        format: Union[Tuple[str,str],str]=  '%(asctime)s {%(filename)20s:%(lineno)3d} p%(process)s %(levelname)s: %(message)s',
        to_stdout=                          True):

    """
    - add_stamp:    prevents merging loggers
    - folder:       writes logfile to folder if given
    - format:       may be given as a str or Tuple[str,str] (fmt,datefmt)
    """

    if not name: name = 'logger'
    if add_stamp: name += '_' + stamp()

    if type(format) is not tuple:
        format = (format,)

    formatter = logging.Formatter(*format)

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
def get_child(
        logger,
        name: Optional[str]=    None,
        change_level: int=      10):
    if not name: name = '_child'
    clogger = logger.getChild(name)
    if change_level != 0:
        lvl = clogger.getEffectiveLevel()
        clogger.setLevel(lvl + change_level)
    return clogger