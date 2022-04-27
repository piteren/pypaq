import numpy as np
import random
import time
import unittest

from pypaq.mpython.mptools import MultiprocParam
from pypaq.mpython.omp import RunningWorker, OMPRunner


class TestOMP(unittest.TestCase):

    def test_OMP_base(self):

        multiproc: MultiprocParam=  10
        n_tasks: int=               50
        max_sec: int=               5

        # basic RunningWorker
        class BRW(RunningWorker):
            def process(self,
                        id: int,
                        sec: int) -> object:
                time.sleep(sec)
                return f'{id}_{sec}'

        ompr = OMPRunner(
            rw_class=       BRW,
            multiproc=      multiproc,
            verb=           1)
        tasks = [{'id': id, 'sec': random.randrange(1, max_sec)} for id in range(n_tasks)]
        results = ompr.process(tasks)

        print(f'({len(results)}) {results}')

        self.assertEqual(len(tasks), len(results))

    # OMPRunner example with process lifetime and exceptions
    def test_OMP_lifetime_exceptions(self):

        multiproc: MultiprocParam=  10
        n_tasks: int=               100
        max_sec: int=               5
        process_lifetime=           2
        exception_prob=             0.1

        # basic RunningWorker
        class BRW(RunningWorker):
            def process(self,
                        id: int,
                        sec: int) -> object:
                if random.random() < exception_prob: raise Exception('randomly crashed')
                time.sleep(sec)
                return f'{id}_{sec}'

        ompr = OMPRunner(
            rw_class=       BRW,
            rw_lifetime=    process_lifetime,
            multiproc=      multiproc,
            verb=           1)

        tasks = [{'id': id, 'sec': random.randrange(1, max_sec)} for id in range(n_tasks)]
        results = ompr.process(tasks, exit=False)
        print(f'({len(results)}) {results}')
        self.assertEqual(n_tasks, len(results))
        self.assertEqual(len(tasks), len(results))

        tasks = [{'id': id, 'sec': random.randrange(1, max_sec)} for id in range(30)] # additional 30 tasks
        results = ompr.process(tasks, exit=False)
        print(f'({len(results)}) {results}')
        self.assertEqual(len(tasks), len(results))

        ompr.exit()

    # example with memory
    def test_OMP_memory(self):

        multiproc: MultiprocParam = 'auto'
        data_size: int = 1000
        n_tasks: int = 1000

        class TMP(RunningWorker):
            def process(self, ds):
                revds = []
                for d in reversed(ds):
                    revds.append(d)
                return revds

        ompr = OMPRunner(
            rw_class=           TMP,
            multiproc=          multiproc,
            verb=               2)

        some_data = [np.random.random(data_size) for _ in range(data_size)] # list of arrays of random floats
        tasks = [{'ds': list(reversed(some_data))} for _ in range(n_tasks)] # tasks with data
        results = ompr.process(tasks)
        print(len(results))
        self.assertEqual(len(tasks), len(results))


if __name__ == '__main__':
    unittest.main()