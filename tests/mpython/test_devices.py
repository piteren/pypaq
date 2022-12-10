import unittest

from pypaq.mpython.devices import get_cuda_mem, get_available_cuda_id, report_cuda, get_devices, mask_cuda, mask_cuda_devices


class TestDevices(unittest.TestCase):

    def test_get_cuda_mem(self):
        mem = get_cuda_mem()
        print(mem)

    def test_get_available_cuda_id(self):
        av_cuda = get_available_cuda_id()
        print(av_cuda)

    def test_report_cuda(self):
        print(report_cuda())

    def test_get_devices_base(self):

        self.assertRaises(NameError, get_devices, namespace='wrong')  # wrong namespace

        self.assertRaises(Exception, get_devices, 'alll') # wrong device

        self.assertRaises(Exception, get_devices, 0.1)  # wrong device

    def test_get_devices_pypaq_representations(self):

        d = get_devices(0, namespace=None)
        print(d)
        self.assertTrue(d==[0])

        d = get_devices(1, namespace=None)
        print(d)
        self.assertTrue(d==[1])

        d = get_devices(13, namespace=None)
        print(d)
        self.assertTrue(d==[13])

        d = get_devices(-1, namespace=None)
        print(d)
        self.assertTrue(type(d) is list and len(d)==1 and ((type(d[0]) is int and d[0]>=0) or type(d) is type(None)))

        d = get_devices([], namespace=None)
        print(d)
        self.assertTrue(list(set([type(e) for e in d]))[0] in [int, None])

        d = get_devices(None, namespace=None)
        print(d)
        self.assertTrue(d==[None])

        d = get_devices('all', namespace=None)
        print(d)
        self.assertTrue(set(d)=={None})

        d = get_devices('/device:CPU:0', namespace=None)
        print(d)
        self.assertTrue(d==[None])

        d = get_devices('GPU:1', namespace=None)
        print(d)
        self.assertTrue(d==[1])

        d = get_devices('cuda:8', namespace=None)
        print(d)
        self.assertTrue(d==[8])

        d = get_devices([0,'all'], namespace=None)
        print(d)
        self.assertTrue(None in d and 0 in d and len(d)>1)

        d = get_devices([0,[],None], namespace=None)
        print(d)
        self.assertTrue(None in d and 0 in d and len(d)>2)

        d = get_devices([0,2,[],None,-1], namespace=None)
        print(d)
        self.assertTrue(None in d and 0 in d and -1 not in d and len(d)>3)

        d = get_devices([1,2,[],None,-1,'all','cuda:0'], namespace=None)
        print(d)
        self.assertTrue(None in d and 0 in d and -1 not in d and len(d)>3)

    def test_get_devices_libraries(self):

        d = get_devices(0, namespace='torch')
        print(d)
        self.assertTrue(d == ['cuda:0'])

        d = get_devices(-1, namespace='torch')
        print(d)
        self.assertTrue(d == ['cuda:1'])

        d = get_devices(None, namespace='torch')
        print(d)
        self.assertTrue(d == ['cpu:0'])

        d = get_devices([0,1,'GPU:0',None], namespace='torch')
        print(d)
        self.assertTrue(d == ['cuda:0', 'cuda:1', 'cuda:0', 'cpu:0'])

    def test_mask_cuda(self):
        av_cuda = get_available_cuda_id()
        print(av_cuda)
        mask_cuda(av_cuda)

    def test_mask_cuda_devices(self):
        d = get_devices([None, 0, 0, '/device:GPU:0', None], namespace='torch')
        print(d)
        mask_cuda_devices(d)
