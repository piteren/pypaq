from multiprocessing import Queue
import random
import time
import unittest

from pypaq.mpython.mpdecor import proc, proc_wait, proc_return, proc_que


class Test_mpdecor(unittest.TestCase):

    def test_proc(self):

        @proc
        def task(to=5):
            for i in range(to):
                print(i, 'done')
                time.sleep(0.1)

        print(task.__name__)
        task(to=10)
        task(10)
        task(10)

        @proc
        def calc(n):
            val = 2 * n
            print(f'will return {val}')
            return val

        print(f'returned {calc(3)}')
        print(f'returned {calc(4)}')


    def test_proc_wait(self):

        @proc_wait
        def task(to=5):
            for i in range(to):
                print(i, 'done')
                time.sleep(0.1)

        task(to=10)
        task(10)
        task(10)


    def test_return(self):

        @proc_return
        def calc(n):
            val = 2 * n
            print(f'will return {val}')
            return val

        print(f'returned {calc(3)}')
        print(f'returned {calc(4)}')


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


    def test_proc_que_more(self):

        que = Queue()

        @proc_que(que)
        def task(name='def', to=5):
            sum = 0
            for i in range(to):
                print(f'calculating sum ({to})...')
                time.sleep(0.5)
                sum += i
            return name, sum

        print(task.__name__)

        n = 50
        for i in range(n): task(name=f'task_{i}', to=random.randrange(5, 20))
        for _ in range(n): print(f'calculated result: {que.get()}')