from copy import deepcopy
from typing import Optional

from pypaq.lipytools.files import prep_folder, r_json, w_json
from pypaq.pms.base import POINT


# Configuration Manager, keeps configuration POINT {key: value}, loads from and saves to file
class ConfigManager:

    def __init__(
            self,
            file: str,
            config: Optional[POINT]=    None,   # {param: value}
            try_to_load=                True):  # tries to load from file if file exists

        self.__file = file
        self.__config: POINT = config if config is not None else {}

        prep_folder(file)

        if try_to_load:
            self.load()

        self.__save_file()

    # loads configuration from file and returns it
    def __read_file(self) -> POINT:
        return r_json(self.__file)

    # saves configuration to jsonl file
    def __save_file(self):
        w_json(self.__config, self.__file)

    # returns (copy of) config
    def get_config(self) -> POINT:
        return deepcopy(self.__config)

    # loads configuration from file, updates self, returns new configuration (from file) or only keys that have changed values
    def load(self, return_only_changed=True) -> POINT:

        file_config = self.__read_file() or {}
        config_changed = {}
        for k in file_config:
            if k not in self.__config or self.__config[k] != file_config[k]:
                self.__config[k] = file_config[k]
                config_changed[k] = file_config[k]

        return config_changed if return_only_changed else file_config

    # updates self with given kwargs, saves file if needed
    def update(
            self,
            return_only_changed=    True,
            **kwargs) -> POINT:

        config_changed = {}
        for k in kwargs:
            if k not in self.__config:
                self.__config[k] = kwargs[k]
                config_changed[k] = kwargs[k]
            else:
                if kwargs[k] != self.__config[k]:
                    self.__config[k] = kwargs[k]
                    config_changed[k] = kwargs[k]

        if config_changed: self.__save_file()

        return config_changed if return_only_changed else self.get_config()