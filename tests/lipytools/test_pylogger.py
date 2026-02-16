from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.pylogger import PyLogger, get_pylogger, get_child

from tests.envy import flush_tmp_dir

TEMP_FD = f'{flush_tmp_dir()}/hpmser'


def setup_function():
    prep_folder(TEMP_FD, flush_non_empty=True)


def test_base():
    logger = get_pylogger()
    assert type(logger) is PyLogger and logger.level == 20 and logger.getEffectiveLevel() == 20
    logger = get_pylogger(level=30)
    assert type(logger) is PyLogger and logger.level == 30 and logger.getEffectiveLevel() == 30


def test_name():
    logger = get_pylogger()
    print(logger.name)
    assert logger.name.startswith('pylogger_')

    logger = get_pylogger(folder=f'{TEMP_FD}/logger')
    print(logger.name)
    assert logger.name.startswith('pylogger_')


def test_flat_child():

    class AU:
        def __init__(self, logger):
            self.logger = logger
            self.logger.info('AU info')
            self.logger.debug('AU debug')

    class A:
        def __init__(self, logger):
            self.logger = logger
            self.logger.info('A info')
            self.logger.debug('A debug')
            self.au = AU(
                logger= get_child(
                    logger=         self.logger,
                    change_level=   10))

    logger = get_pylogger(level=10, flat_child=True)
    ab = A(logger)
    print(ab.logger.level)
    print(ab.au.logger.getEffectiveLevel())
    assert ab.logger.level == 10 and ab.au.logger.getEffectiveLevel() == 10
