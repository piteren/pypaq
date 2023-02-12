import unittest

from pypaq.pms.base import POINT, point_str, get_params, point_trim


class TestPOINT(unittest.TestCase):

    def test_point_str(self):
        p: POINT = {
            'value':        1234.5,
            'axis_one':     15.4,
            'name':         'sample_point',
            'drop':         0.0,
            'baseLR':          1.5e-6}
        ps = point_str(p)
        self.assertEqual(sum([1 for k in p if k in ps]), len(p))
        print(point_str(p))


    def test_get_params(self):

        def func_a(
                *args,
                a,
                b,
                c=10,
                d=True,
                **kwargs):
            print(a,b,c,d)

        def func_b(
                a,
                b,
                c=10,
                d=True):
            print(a,b,c,d)

        def func_c(
                a,
                b):
            print(a,b)

        def func_d(
                c=10,
                d=True):
            print(c,d)

        print(get_params(func_a))
        self.assertTrue(get_params(func_a) == {'without_defaults': ['a', 'b'], 'with_defaults': {'c': 10, 'd': True}})

        print(get_params(func_b))
        self.assertTrue(get_params(func_b) == {'without_defaults': ['a', 'b'], 'with_defaults': {'c': 10, 'd': True}})

        print(get_params(func_c))
        self.assertTrue(get_params(func_c) == {'without_defaults': ['a', 'b'], 'with_defaults': {}})

        print(get_params(func_d))
        self.assertTrue(get_params(func_d) == {'without_defaults': [], 'with_defaults': {'c': 10, 'd': True}})


    def test_point_trim(self):

        def func(
                a,
                b,
                c=10,
                d=True):
            print(a,b,c,d)

        point_wide = {'a':3, 'b':5, 'g':11, 'aa':0}

        print(point_trim(func, point_wide))
        self.assertTrue(point_trim(func, point_wide) == {'a': 3, 'b': 5})