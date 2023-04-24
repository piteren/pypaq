import time
import unittest

from tests.envy import flush_tmp_dir

from pypaq.lipytools.files import prep_folder
from pypaq.pms.config_manager import ConfigManager

CONFIG_DIR = f'{flush_tmp_dir()}/config_manager'


class TestCM(unittest.TestCase):

    def setUp(self) -> None:
        prep_folder(CONFIG_DIR, flush_non_empty=True)

    # you should go to {TEMP_DIR}/config.file and edit>save it while running this test
    def test_base(self):

        n_loops = 6

        config = {
            'param_aaa':    15,
            'beta':         20.45,
            'do_it':        False,
            'dont_do':      None}

        cm = ConfigManager(f'{CONFIG_DIR}/config.file', config=config)
        print(cm.get_config())

        for _ in range(n_loops):
            time.sleep(5)
            cm.load()
            print(cm.get_config())

        cm = ConfigManager(file_FP=f'{CONFIG_DIR}/config.file')
        print(cm.get_config())

    def test_base_set(self):

        # saves sample config file
        config = {
            'param_aaa':    15,
            'beta':         20.45,
            'do_it':        False,
            'dont_do':      None}
        cm = ConfigManager(f'{CONFIG_DIR}/config.file', config=config, loglevel=30)
        print(cm)
        self.assertTrue(cm.param_aaa == 15 and cm.beta == 20.45)

        cm = ConfigManager(f'{CONFIG_DIR}/config.file')
        print(cm)

        cm.alfa = 13
        cm.beta = 19.99
        cm['gamma'] = 14
        print(cm)
        self.assertTrue(cm.alfa == 13 and cm.beta == 19.99 and cm.gamma == 14)