import logging
import unittest

from pypaq.lipytools.pylogger import get_pylogger, get_child


class TestPylogger(unittest.TestCase):

    def test_base(self):
        logger = get_pylogger()
        self.assertTrue(type(logger) is logging.Logger)

    def test_flat_child(self):

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
        self.assertTrue(ab.logger.level == 10 and ab.au.logger.getEffectiveLevel() == 10)