import time
import unittest

from tests.envy import get_tmp_dir

from pypaq.lipytools.little_methods import prep_folder
from pypaq.pms.config_manager import ConfigManager

CONFIG_DIR = f'{get_tmp_dir()}/config_manager'


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
            newc = cm.load()
            print(newc, cm.get_config())

        cm = ConfigManager(file=f'{CONFIG_DIR}/config.file')
        print(cm.get_config())


if __name__ == '__main__':
    unittest.main()