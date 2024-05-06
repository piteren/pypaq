from typing import Optional

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.files import prep_folder, r_json, w_json
from pypaq.pms.base import POINT


# Configuration Manager, keeps configuration in attributes of self, manages config file (json)
class ConfigManager:

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
        self._logger.info(f'*** ConfigManager *** initializes, config file: {self._file}')

        _conf = {}

        if config_init is not None:
            self._logger.info(f'> got initial config: {config_init}')
            _conf.update(config_init)

        file_config = r_json(self._file)
        if file_config:
            self._logger.info(f'> got file config:    {file_config}')
            if override_with_saved:
                _conf.update(file_config)

        self.__config = _conf
        w_json(_conf, file_FP)

    # sets attribute
    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:

            # first update from file
            file_config = r_json(self._file)
            self.__config.update(file_config)

            new_attribute = key not in self.__config
            new_value = not new_attribute and self.__config[key] != value
            if new_attribute or new_value:
                self.__config[key] = value
                if new_attribute:   self._logger.info(f'set new attribute: {key}: {value}')
                if new_value:       self._logger.info(f'set new value:     {key}: {value}')

            w_json(self.__config, self._file)

    # returns attribute value
    def __getattribute__(self, name):
        if name.startswith('_') or name in ['update_config','get_config']:
            return super().__getattribute__(name)
        else:

            # update from file
            file_config = r_json(self._file)
            changed_sth = False
            for k in file_config:
                if k not in self.__config or k in self.__config and self.__config[k] != file_config[k]:
                    self.__config[k] = file_config[k]
                    changed_sth = True

            if changed_sth:
                w_json(self.__config, self._file)

            return self.__config[name]

    # update with a given dictionary
    def update_config(self, dct:POINT) -> None:

        # first update from file
        file_config = r_json(self._file)
        self.__config.update(file_config)

        if dct:
            self.__config.update(dct)
            self._logger.info(f'updated configuration with: {dct}')

        w_json(self.__config, self._file)

    def get_config(self) -> POINT:
        self.update_config({}) # to update from file
        c = {}
        c.update(self.__config)
        return c

    def __str__(self):
        return str(self.get_config())