from multiprocessing import Queue
import random
import time
import unittest

from pypaq.mpython.mpdecor import proc, proc_wait, proc_return, proc_que


class Test_mpdecor(unittest.TestCase):

    def test_proc(self):
        # TODO: write test
        pass

    def test_proc_wait(self):
        # TODO: write test
        pass

    def test_return(self):
        # TODO: write test
        pass

    def test_proc_que(self):

        n_tasks = 10

        que = Queue()

        @proc_que(que)
        def task():
            sl = random.randint(3,8)
            print(f'sleeping for {sl}')
            time.sleep(sl)
            return sl

        s_time = time.time()
        for r in range(n_tasks):
            task()

        res = [que.get() for _ in range(n_tasks)]
        t_time = time.time() - s_time
        print(res)

        self.assertTrue(len(res) == n_tasks)
        self.assertTrue(res[0]<=res[-1])
        self.assertTrue(res[-1] <= t_time <= res[-1]+1)