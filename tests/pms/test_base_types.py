import unittest

from pypaq.pms.base_types import POINT, point_str


class TestPOINT(unittest.TestCase):
    #"""
    def test_point_str(self):
        p: POINT = {
            'value':        1234.5,
            'axis_one':     15.4,
            'name':         'sample_point',
            'drop':         0.0,
            'iLR':          1.5e-6}
        ps = point_str(p)
        self.assertEqual(sum([1 for k in p if k in ps]), len(p))
        print(point_str(p))
    #"""

if __name__ == '__main__':
    unittest.main()