import unittest

from pypaq.textools.text_processing import whitespace_normalization


class TestTextProcessing(unittest.TestCase):

    def test_whitespace_normalization(self):
        for t,r in [
            ('', ''),
            (' ', ''),
            ('\n', ''),
            ('a\na', 'a a'),
            ('a\n  a', 'a a'),
            ('a\n  \n\na', 'a a'),
        ]:
            rp = whitespace_normalization(t)
            print(f'>{rp}<->{r}<')
            self.assertTrue(rp == r)