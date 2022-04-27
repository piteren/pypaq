"""

 2018 (c) piteren

"""

import os
import sys
from typing import Optional

from pypaq.lipytools.little_methods import prep_folder, stamp


# logger duplicating print() output to given file
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