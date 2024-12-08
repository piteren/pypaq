import unittest

from pypaq.mpython.devices import get_devices


class TestDevices(unittest.TestCase):

    def test_base(self):

        rd = get_devices(devices='all')
        n_all = len(rd)

        for d,c in [
            (None,                  1),
            (0.5,                   round(0.5*n_all)),
            ('all',                 n_all),
            ('cpu',                 1),
            ([None,None],           2),
            ([0.3,None],            round(0.3*n_all)+1),
            (['cpu',None,'cpu'],    3),
            (['all','cpu',None],    n_all+2),
            ([1.0],                 n_all),
            ([0.5,0.5,None],        2*round(0.5*n_all)+1)
        ]:
            rd = get_devices(devices=d)
            print(d,rd)
            self.assertTrue(c==len(rd))
            self.assertTrue(set(rd)=={None})

    def test_raise(self):

        for d in [
            'single',
            1,
            2,
            True,
            [],
            'CPU',
            'ALL',
        ]:
            self.assertRaises(Exception, get_devices, d)

