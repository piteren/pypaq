import logging
import unittest

from pypaq.lipytools.pylogger import get_pylogger


class TestPylogger(unittest.TestCase):

    def test_base(self):
        logger = get_pylogger()
        self.assertTrue(type(logger) is logging.Logger)