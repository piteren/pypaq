import logging
from typing import Optional, Union, Tuple

from pypaq.lipytools.printout import stamp
from pypaq.lipytools.files import prep_folder


def get_pylogger(
        name: Optional[str]=                None,
        add_stamp=                          True,
        folder: Optional[str]=              None,
        level=                              logging.INFO,
        flat_child: bool=                   False,
        format: Union[Tuple[str,str],str]=  '%(asctime)s {%(filename)20s:%(lineno)4d} p%(process)s %(levelname)s: %(message)s',
        to_stdout=                          True,
) -> logging.Logger:
    """
    # returns formatted logging.Logger
    - add_stamp:    prevents merging loggers of same name
    - folder:       writes logfile to folder if given
    - flat_child:   forces child of this logger created with get_child() to be same level
    - format:       may be given as a str or Tuple[str,str] (fmt,datefmt)
    """

    if not name and folder:
        name = 'pylogger'
    if name and add_stamp:
        name += '_' + stamp()

    if type(format) is not tuple:
        format = (format,)

    formatter = logging.Formatter(*format)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    ### manage handlers

    # if logger has handlers new handlers of existing type won't be added
    for h in logger.handlers:
        if type(h) is logging.FileHandler:
            folder = None
        if type(h) is logging.StreamHandler:
            to_stdout = False

    if folder:
        prep_folder(folder)
        fh = logging.FileHandler(f'{folder}/{name}.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if to_stdout:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    logger.flat_child = flat_child

    return logger

# returns child with optionally changed level
def get_child(
        logger,
        name: Optional[str]=    None,
        change_level: int=      10,
) -> logging.Logger:

    if not name:
        name = '_child'

    clogger = logger.getChild(name)
    clogger.flat_child = logger.flat_child

    if change_level != 0 and not logger.flat_child:
        lvl = clogger.getEffectiveLevel()
        lvl_child = min(lvl + change_level, logging.WARNING)
        clogger.setLevel(lvl_child)

    return clogger