from typing import Optional

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.files import prep_folder, r_json, w_json
from pypaq.pms.base import POINT
from pypaq.pms.subscriptable import Subscriptable


# Configuration Manager, keeps configuration in attributes of self - accessible as fields or dict style, manages config file (json)
class ConfigManager(Subscriptable):

    def __init__(
            self,
            file_FP: str,                           # full path to config file
            config_init: Optional[POINT]=   None,
            override_with_saved: bool=      True,   # file (if exists) will override given initial config
            logger=                         None,
            loglevel=                       20):

        if not logger:
            logger = get_pylogger(name='ConfigManager', level=loglevel)
        self._logger = logger

        self._file = file_FP
        prep_folder(self._file)
        self._logger.info(f'*** ConfigManager *** inits, config file: {self._file}')

        file_config = r_json(self._file)
        if file_config:
            self._logger.info(f'> got file config:    {file_config}')

        if config_init is not None:
            self._logger.info(f'> got initial config: {config_init}')

        start_config = {}
        if config_init:
            start_config.update(config_init)

        if override_with_saved and file_config:
            start_config.update(file_config)

        self.update(**start_config)

    # saves configuration to file
    def __save_file(self):
        w_json(self.get_config(), self._file)

    # (eventually) loads new configuration from file
    def load(self):
        file_config = r_json(self._file)
        if file_config is not None:
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

    # updates self (dict like update) with given kwargs
    def update(self, **kwargs):
        super().update(dct=kwargs)

    # alias to get_point of Subscriptable
    def get_config(self) -> POINT:
        return self.get_point()