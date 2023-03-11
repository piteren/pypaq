from copy import deepcopy
from typing import Optional

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.files import prep_folder, r_json, w_json
from pypaq.pms.base import POINT


# Configuration Manager, keeps configuration POINT {key: value}, loads from and saves to file (json)
class ConfigManager:

    def __init__(
            self,
            file_FP: str,                       # full path to config file
            config: Optional[POINT]=    None,   # {param: value}
            try_to_load=                True,   # tries to load from file if file exists
            logger=                     None,
            loglevel=                   20):

        if not logger:
            logger = get_pylogger(name='ConfigManager', level=loglevel)
        self.logger = logger

        self.logger.info(f'*** ConfigManager *** inits, file: {file_FP}')

        self.__file = file_FP
        self.__config: POINT = {}
        if config:
            self.logger.info(f'> initial config: {config}')
            self.__config = deepcopy(config)

        prep_folder(file_FP)

        if try_to_load:
            self.load()

        self.__save_file()

    # saves configuration to file
    def __save_file(self):
        w_json(self.__config, self.__file)

    # returns (deepcopy of) config
    def get_config(self) -> POINT:
        return deepcopy(self.__config)

    # loads configuration from file
    def load(self) -> POINT:

        file_config = r_json(self.__file) or {}
        config_changed = {}
        for k in file_config:
            if k not in self.__config or self.__config[k] != file_config[k]:
                self.__config[k] = file_config[k]
                config_changed[k] = file_config[k]

        if config_changed:
            self.logger.info(f'> loaded new config changes: {config_changed}')

        return self.get_config()

    # updates self with given kwargs, saves file if needed
    def update(self, **kwargs) -> POINT:

        config_changed = {}
        for k in kwargs:
            if k not in self.__config:
                self.__config[k] = kwargs[k]
                config_changed[k] = kwargs[k]
            else:
                if kwargs[k] != self.__config[k]:
                    self.__config[k] = kwargs[k]
                    config_changed[k] = kwargs[k]

        if config_changed:
            self.__save_file()
            self.logger.info(f'> saved new config changes: {config_changed}')

        return self.get_config()

    # alias to get_config()
    def __call__(self, *args, **kwargs):
        return self.get_config()