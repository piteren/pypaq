import unittest

from pypaq.mpython.devices import get_cuda_mem, get_available_cuda_id, report_cuda, get_devices, mask_cuda, mask_cuda_devices


class TestMPTools(unittest.TestCase):

    def test_get_cuda_mem(self):
        mem = get_cuda_mem()
        print(mem)

    def test_get_available_cuda_id(self):
        av_cuda = get_available_cuda_id()
        print(av_cuda)

    def test_report_cuda(self):
        report_cuda()

    def test_get_devices(self):

        devs = get_devices(0)
        print(devs)
        self.assertTrue(len(devs) == 1)
        self.assertTrue(devs[0] == '/device:GPU:0')

        dev = get_devices(-1)
        print(dev)
        self.assertTrue(len(devs) == 1)
        self.assertTrue('/device:GPU' in dev[0])

        devs = get_devices(0, tf2_naming=True)
        print(devs)
        self.assertTrue(len(devs) == 1)
        self.assertTrue(devs[0] == 'GPU:0')

        devs = get_devices(None, tf2_naming=True)
        print(devs)
        self.assertTrue(len(devs) == 1)
        self.assertTrue(devs[0] == 'CPU:0')

        devs = get_devices('all', tf2_naming=True)
        print(devs)
        self.assertTrue(len(devs) > 0)
        self.assertTrue(list(set(devs))[0] == 'CPU:0')

        devs = get_devices('/device:GPU:0', tf2_naming=True)
        print(devs)
        self.assertTrue(len(devs) == 1)
        self.assertTrue(devs[0] == 'GPU:0')

        devs = get_devices('GPU:1', tf2_naming=True)
        print(devs)
        self.assertTrue(len(devs) == 1)
        self.assertTrue(devs[0] == 'GPU:1')

        devs = get_devices('GPU:1', tf2_naming=False)
        print(devs)
        self.assertTrue(len(devs) == 1)
        self.assertTrue(devs[0] == '/device:GPU:1')

        devs = get_devices(['GPU:0', '/device:GPU:1'], tf2_naming=True)
        print(devs)
        self.assertTrue(devs[0] == 'GPU:0' and devs[1] == 'GPU:1')

        devs = get_devices([], tf2_naming=True)
        print(devs)
        for d in devs: self.assertTrue('GPU' in d)

        devs = get_devices([0]*5, tf2_naming=True)
        print(devs)
        self.assertTrue(len(devs) == 5)
        for d in devs: self.assertTrue(d == 'GPU:0')

        devs = get_devices([None]*5, tf2_naming=True)
        print(devs)
        self.assertTrue(len(devs) == 5)
        for d in devs: self.assertTrue(d == 'CPU:0')

        devs = get_devices([None, 0, 0, '/device:GPU:0', None], tf2_naming=True)
        print(devs)
        self.assertTrue(devs == ['CPU:0', 'CPU:0', 'GPU:0', 'GPU:0', 'GPU:0'])

    def test_mask_cuda(self):
        av_cuda = get_available_cuda_id()
        print(av_cuda)
        mask_cuda(av_cuda)

    def test_mask_cuda_devices(self):
        devs = get_devices([None, 0, 0, '/device:GPU:0', None], tf2_naming=True)
        print(devs)
        mask_cuda_devices(devs, verb=1)
