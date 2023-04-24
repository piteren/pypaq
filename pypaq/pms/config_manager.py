from typing import Optional

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.files import prep_folder, r_json, w_json
from pypaq.pms.base import POINT
from pypaq.pms.subscriptable import Subscriptable


# Configuration Manager, keeps configuration in attributes of self - accessible as fields or dict style, manages config file (json)
class ConfigManager(Subscriptable):

    def __init__(
            self,
            file_FP: str,                       # full path to config file
            try_to_load: bool=          True,   # tries to load from file if file exists
            config: Optional[POINT]=    None,   # {param: value}, overrides config from file if exists
            logger=                     None,
            loglevel=                   20):

        if not logger:
            logger = get_pylogger(name='ConfigManager', level=loglevel)
        self._logger = logger

        self._logger.info(f'*** ConfigManager *** inits, file: {file_FP}')

        self._file = file_FP
        prep_folder(file_FP)

        # first try to load from file
        if try_to_load and config is None:
            self.load()

        if config is not None:
            self._logger.info(f'> setting initial config: {config}')
            self.update(**config)

    # saves configuration to file
    def __save_file(self):
        w_json(self.get_config(), self._file)

    # (eventually) loads new configuration from file
    def load(self):
        file_config = r_json(self._file)
        if file_config is not None:
            self._logger.info(f'> loading config from file..')
            super().update(dct=file_config)

    # updates attribute and saves file
    def __setattr__(self, key, value):
        new_attribute = key not in self
        new_value = not new_attribute and self[key] != value
        if new_attribute or new_value:
            super().__setattr__(key, value)
            if not key.startswith('_'):
                if new_attribute:   self._logger.info(f'set new attribute: {key}: {value}')
                if new_value:       self._logger.info(f'set new value:     {key}: {value}')
                self.__save_file()

    # alias to get_point of Subscriptable
    def get_config(self) -> POINT:
        return self.get_point()

    # updates self (dict like update) with given kwargs
    def update(self, **kwargs):
        super().update(dct=kwargs)