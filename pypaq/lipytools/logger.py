"""

 2018 (c) piteren

"""

import logging
import os
import sys
from typing import Optional

from pypaq.lipytools.little_methods import stamp
from pypaq.lipytools.files import prep_folder


# logger duplicates print() output to given file
class Logger:

    def __init__(
            self,
            file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()

    # only to pass unittests with objects that uses logger
    def getvalue(self): return 0


class Logg:

    ERROR =     logging.ERROR   # 0
    WARNING =   logging.WARNING # 0
    INFO =      logging.INFO    # 1
    DEBUG =     logging.DEBUG   # 2

    def __init__(
            self,
            name=                   'logger',
            folder: Optional[str]=  None,
            level=                  None):
        if not level: level = logging.INFO
        filename = f'{folder}/{name}.log' if folder else None
        self.logger = logging
        self.logger.basicConfig(
            level=      level,
            format=     '%(message)s',
            filename=   filename)
        fni = f', file: {filename}' if filename else ''
        self(f'Logger {name} started, level: {self.logger.getLevelName(level=level)}{fni}')


    def __call__(self, msg:str, level=None):
        if level == Logg.DEBUG:             self.logger.debug(msg)
        if level == Logg.INFO or not level: self.logger.info(msg)
        if level == Logg.WARNING:           self.logger.warning(f'WARNING: {msg}')
        if level == Logg.ERROR:             self.logger.error(f'ERROR: {msg}')


# method setting logger to logfile, returns path to logfile
def set_logger(
        log_folder: str,
        custom_name: Optional[str]= None,
        add_stamp=                  True,
        verb=                       1) -> str:

    prep_folder(log_folder)

    file_name = custom_name if custom_name else 'run'
    if add_stamp: file_name += f'_{stamp()}'
    file_name += '.log'

    logfile_path = os.path.join(log_folder, file_name)
    sys.stdout = Logger(logfile_path)

    if verb>0:
        script_path = os.path.join(os.path.dirname(sys.argv[0]), os.path.basename(sys.argv[0]))
        print(f'\nLogger started..')
        print(f' > for script : {script_path}')
        print(f' > logfile    : {logfile_path}')

    return logfile_path