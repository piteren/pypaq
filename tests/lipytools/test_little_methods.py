import unittest

from pypaq.lipytools.little_methods import stamp, print_nested_dict


class TestStamp(unittest.TestCase):

    def test_stamp(self):
        print(stamp())
        print(stamp(letters=None))
        print(stamp(year=True))

    def test_print_nested_dict(self):
        dc = {
            'a0': {
                'a1': {
                    'a2': ['el1','el2']
                }
            },
            'b0': ['el1','el2','el3']
        }
        print_nested_dict(dc)



if __name__ == '__main__':
    unittest.main()