import unittest

import torch

from pypaq.mpython.devices import get_cuda_mem, get_available_cuda_id, report_cuda, _get_devices_pypaq, get_devices, mask_cuda, mask_cuda_devices


class TestDevices(unittest.TestCase):

    def test_get_cuda_mem(self):
        mem = get_cuda_mem()
        print(mem)


    def test_get_available_cuda_id(self):
        av_cuda = get_available_cuda_id()
        print(av_cuda)


    def test_report_cuda(self):
        print(report_cuda())


    def test_get_devices_pypaq(self):

        d = _get_devices_pypaq(0)
        print(d)
        self.assertTrue(d == [0])

        d = _get_devices_pypaq(1)
        print(d)
        self.assertTrue(d == [1])

        d = _get_devices_pypaq(13)
        print(d)
        self.assertTrue(d == [13])

        d = _get_devices_pypaq(-1)
        print(d)
        self.assertTrue(type(d) is list and (not d or len(d) == 1 and ((type(d[0]) is int and d[0] >= 0) or d[0] is None)))

        d = _get_devices_pypaq([])
        print(d)
        self.assertTrue(d == [] or list(set([type(e) for e in d]))[0] in [int, None])

        d = _get_devices_pypaq(None)
        print(d)
        self.assertTrue(d == [None])

        d = _get_devices_pypaq(0.7)
        print(len(d), d)
        self.assertTrue(type(d) is list and len(d)>=1 and d[0] is None)

        d = _get_devices_pypaq(-0.5)
        print(d)
        self.assertTrue(d == [None])

        d = _get_devices_pypaq('all')
        print(len(d), d)
        self.assertTrue(type(d) is list and len(d) >= 1 and d[0] is None)

        d = _get_devices_pypaq('cuda')
        print(d)
        self.assertTrue(d == [0])

        d = _get_devices_pypaq('cuda:0')
        print(d)
        self.assertTrue(d == [0])

        d = _get_devices_pypaq('cuda:1')
        print(d)
        self.assertTrue(d == [1])

        d = torch.device('cpu')
        d = _get_devices_pypaq(d)
        print(d)
        self.assertTrue(d == [None])

        d = torch.device('cuda')
        d = _get_devices_pypaq(d)
        print(d)
        self.assertTrue(d == [0])

        d = torch.device('cuda:1')
        d = _get_devices_pypaq(d)
        print(d)
        self.assertTrue(d == [1])

        d = torch.device('cuda:8')
        d = _get_devices_pypaq(d)
        print(d)
        self.assertTrue(d == [8])

        d = _get_devices_pypaq([0, 'all'])
        print(d)
        self.assertTrue(None in d and 0 in d and len(d)>1)

        d = _get_devices_pypaq([0, [], None])
        print(len(d), d)
        self.assertTrue(None in d and 0 in d and len(d)>1)

        d = _get_devices_pypaq([0, 2, [], None, -1])
        print(len(d), d)
        self.assertTrue(None in d and 0 in d and -1 not in d and len(d)>1)

        d = _get_devices_pypaq([1, 2, [], None, -1, 'all', 'cuda:0'])
        print(len(d), d)
        self.assertTrue(None in d and 0 in d and -1 not in d and len(d)>3)


    def test_get_devices_exceptions(self):
        self.assertRaises(Exception, get_devices, 'alll') # wrong device
        self.assertRaises(Exception, get_devices, (0,1))  # wrong device


    def test_get_devices_torch(self):

        d = get_devices(0)
        print(d)
        self.assertTrue(d == ['cuda:0'])

        d = get_devices(-1)
        print(d)
        self.assertTrue(d == ['cuda:1'] or d == [])

        d = get_devices([])
        print(d)
        self.assertTrue(type(d) is list and (not d or type(d[0]) is str))

        d = get_devices(None)
        print(d)
        self.assertTrue(d == ['cpu'])

        d = get_devices([0,1,'cuda:0',None])
        print(d)
        self.assertTrue(d == ['cuda:0', 'cuda:1', 'cuda:0', 'cpu'])

    def test_mask_cuda(self):
        av_cuda = get_available_cuda_id()
        print(av_cuda)
        mask_cuda(av_cuda)

    def test_mask_cuda_devices(self):
        d = get_devices([None, 0, 0, 'cuda', None], torch_namespace=True)
        print(d)
        mask_cuda_devices(d)
