import unittest

import torch

from pypaq.mpython.devices import get_cuda_mem, get_available_cuda_id, report_cuda, to_dev_pypaq_base, get_devices, mask_cuda, mask_cuda_devices


class TestDevices(unittest.TestCase):

    def test_get_cuda_mem(self):
        mem = get_cuda_mem()
        print(mem)


    def test_get_available_cuda_id(self):
        av_cuda = get_available_cuda_id()
        print(av_cuda)


    def test_report_cuda(self):
        print(report_cuda())


    def test_to_dev_pypaq_base(self):

        d = to_dev_pypaq_base(0)
        print(d)
        self.assertTrue(d == [0])

        d = to_dev_pypaq_base(1)
        print(d)
        self.assertTrue(d == [1])

        d = to_dev_pypaq_base(13)
        print(d)
        self.assertTrue(d == [13])

        d = to_dev_pypaq_base(-1)
        print(d)
        self.assertTrue(type(d) is list and len(d) == 1 and ((type(d[0]) is int and d[0] >= 0) or type(d) is type(None)))

        d = to_dev_pypaq_base([])
        print(d)
        self.assertTrue(list(set([type(e) for e in d]))[0] in [int, None])

        d = to_dev_pypaq_base(None)
        print(d)
        self.assertTrue(d == [None])

        d = to_dev_pypaq_base(0.7)
        print(d)
        self.assertTrue(type(d) is list and len(d)>=1 and d[0] is None)

        d = to_dev_pypaq_base(-0.5)
        print(d)
        self.assertTrue(d == [None])

        d = to_dev_pypaq_base('all')
        print(d)
        self.assertTrue(type(d) is list and len(d) >= 1 and d[0] is None)

        d = to_dev_pypaq_base('cpu')
        print(d)
        self.assertTrue(d == [None])

        d = to_dev_pypaq_base('cuda')
        print(d)
        self.assertTrue(d == [0])

        d = to_dev_pypaq_base('cuda:0')
        print(d)
        self.assertTrue(d == [0])

        d = to_dev_pypaq_base('cuda:1')
        print(d)
        self.assertTrue(d == [1])

        d = torch.device('cpu')
        d = to_dev_pypaq_base(d)
        print(d)
        self.assertTrue(d == [None])

        d = torch.device('cuda')
        d = to_dev_pypaq_base(d)
        print(d)
        self.assertTrue(d == [0])

        d = torch.device('cuda:1')
        d = to_dev_pypaq_base(d)
        print(d)
        self.assertTrue(d == [1])

        d = torch.device('cuda:8')
        d = to_dev_pypaq_base(d)
        print(d)
        self.assertTrue(d == [8])

        d = to_dev_pypaq_base([0,'all'])
        print(d)
        self.assertTrue(None in d and 0 in d and len(d)>1)

        d = to_dev_pypaq_base([0,[],None])
        print(d)
        self.assertTrue(None in d and 0 in d and len(d)>2)

        d = to_dev_pypaq_base([0,2,[],None,-1])
        print(d)
        self.assertTrue(None in d and 0 in d and -1 not in d and len(d)>3)

        d = to_dev_pypaq_base([1,2,[],None,-1,'all','cuda:0'])
        print(d)
        self.assertTrue(None in d and 0 in d and -1 not in d and len(d)>3)


    def test_get_devices_base(self):
        self.assertRaises(Exception, get_devices, 'alll') # wrong device
        self.assertRaises(Exception, get_devices, (0,1))  # wrong device


    def test_get_devices_torch(self):

        d = get_devices(0)
        print(d)
        self.assertTrue(d == ['cuda:0'])

        d = get_devices(-1)
        print(d)
        self.assertTrue(d == ['cuda:1'])

        d = get_devices([])
        print(d)
        self.assertTrue(type(d) is list and type(d[0]) is str)

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
