import logging
import os
from pathlib import Path

import pytest

from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.pylogger import logging_mod

TEMP_FD = Path(__file__).parent / '_tmp_pylogger'


@pytest.fixture(autouse=True, scope='module')
def tmp_dir():
    prep_folder(TEMP_FD, flush_non_empty=True)


@pytest.fixture(autouse=True)
def reset_root_logger():
    yield
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
    root.handlers.clear()


# logging_mod

def test_logging_mod_adds_stream_handler():
    logging_mod()
    handlers = [h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler]
    assert len(handlers) == 1


def test_logging_mod_stream_idempotent():
    logging_mod()
    logging_mod()
    handlers = [h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler]
    assert len(handlers) == 1


def test_logging_mod_level():
    logging_mod(level=logging.DEBUG)
    assert logging.getLogger().level == logging.DEBUG


def test_logging_mod_file_handler():
    logging_mod(folder=str(TEMP_FD), log_file_name='test_app')
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert os.path.exists(f'{TEMP_FD}/test_app.log')


def test_logging_mod_file_not_idempotent():
    logging_mod(folder=str(TEMP_FD), log_file_name='app_a')
    logging_mod(folder=str(TEMP_FD), log_file_name='app_b')
    file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 2