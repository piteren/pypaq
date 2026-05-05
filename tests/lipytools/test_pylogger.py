import logging
import os
from pathlib import Path

import pytest

from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.pylogger import setup_logging, Logged

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


# setup_logging

def test_setup_logging_adds_stream_handler():
    setup_logging()
    handlers = [h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler]
    assert len(handlers) == 1


def test_setup_logging_stream_idempotent():
    setup_logging()
    setup_logging()
    handlers = [h for h in logging.getLogger().handlers if type(h) is logging.StreamHandler]
    assert len(handlers) == 1


def test_setup_logging_level():
    setup_logging(level=logging.DEBUG)
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_file_handler():
    setup_logging(folder=str(TEMP_FD), log_file_name='test_app')
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert os.path.exists(f'{TEMP_FD}/test_app.log')


def test_setup_logging_file_not_idempotent():
    setup_logging(folder=str(TEMP_FD), log_file_name='app_a')
    setup_logging(folder=str(TEMP_FD), log_file_name='app_b')
    file_handlers = [h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 2


# Logged.get_logger

def test_get_logger_class_name():
    class MyWorker(Logged):
        pass
    w = MyWorker()
    assert w.get_logger().name == f'{MyWorker.__module__}.{MyWorker.__qualname__}'


def test_get_logger_object_name():
    class MyWorker(Logged):
        pass
    w = MyWorker()
    w.name = 'worker1'
    assert w.get_logger().name == f'{MyWorker.__module__}.{MyWorker.__qualname__}.worker1'


def test_get_logger_level():
    class MyWorker(Logged):
        pass
    w = MyWorker()
    assert w.get_logger(level=logging.WARNING).level == logging.WARNING


def test_get_logger_file_handler():
    class MyWorker(Logged):
        pass
    w = MyWorker()
    w.name = 'file_worker'
    logger = w.get_logger(folder=str(TEMP_FD))
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
    assert os.path.exists(f'{TEMP_FD}/file_worker.log')


def test_get_logger_file_handler_idempotent():
    class MyWorker(Logged):
        pass
    w = MyWorker()
    w.name = 'idempotent_worker'
    w.get_logger(folder=str(TEMP_FD))
    logger = w.get_logger(folder=str(TEMP_FD))
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1


def test_get_logger_two_objects_separate_loggers():
    class MyWorker(Logged):
        pass
    w1, w2 = MyWorker(), MyWorker()
    w1.name = 'alpha'
    w2.name = 'beta'
    assert w1.get_logger() is not w2.get_logger()
    assert w1.get_logger().name != w2.get_logger().name