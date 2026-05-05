import time
from pathlib import Path
import pytest

from pypaq.lipytools.files import prep_folder
from pypaq.pms.config_manager import ConfigManager

CONFIG_DIR = Path(__file__).parent / '_tmp_config_manager'


@pytest.fixture(autouse=True, scope='module')
def tmp_dir():
    prep_folder(CONFIG_DIR, flush_non_empty=True)


def test_base_set():

    # saves sample config file
    config = {
        'param_aaa':    15,
        'beta':         20.45,
        'do_it':        False,
        'dont_do':      None}
    cm = ConfigManager(f'{CONFIG_DIR}/config.file', config_init=config, loglevel=30)
    print(cm)
    assert cm.param_aaa == 15 and cm.beta == 20.45

    cm = ConfigManager(f'{CONFIG_DIR}/config.file')
    print(cm)

    cm.alfa = 13
    cm.beta = 19.99
    cm.gamma = 14
    print(cm)
    assert cm.alfa == 13 and cm.beta == 19.99 and cm.gamma == 14

    cm.update_config({'something':111})
    print(cm)
    assert cm.get_config() == {'param_aaa': 15, 'beta': 19.99, 'do_it': False, 'dont_do': None, 'alfa': 13, 'gamma': 14, 'something': 111}
