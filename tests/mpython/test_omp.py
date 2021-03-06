import random
import time
import unittest

from pypaq.mpython.mptools import MultiprocParam
from pypaq.mpython.omp_blocking import RunningWorker
from pypaq.mpython.omp import OMPRunner


# basic RunningWorker with random exception
class BRW(RunningWorker):
    def process(
            self,
            id: int,
            sec: int,
            exception_prob: float=  0.0) -> object:
        if random.random() < exception_prob: raise Exception('randomly crashed')
        time.sleep(sec)
        return f'{id}_{sec}'


class TestOMP_NB(unittest.TestCase):

    def test_OMP_NB_base(self):

        multiproc: MultiprocParam=  10
        n_tasks: int=               50
        max_sec: int=               5

        ompr = OMPRunner(
            rw_class=       BRW,
            multiproc=      multiproc,
            verb=           1)
        tasks = [{
            'id':   id,
            'sec':  random.randrange(1, max_sec)}
            for id in range(n_tasks)]

        ompr.process(tasks)
        results = ompr.get_all_results()

        print(f'({len(results)}) {results}')

        self.assertTrue(isinstance(results[0], str))
        self.assertEqual(len(tasks), len(results))

        results = ompr.get_all_results()
        self.assertTrue(results == [])

        ompr.exit()

    # OMPRunner example with process lifetime and exceptions
    def test_OMP_NB_lifetime_exceptions(self):

        multiproc: MultiprocParam=  10
        n_tasks: int=               100
        max_sec: int=               5
        process_lifetime=           2
        exception_prob=             0.1

        ompr = OMPRunner(
            rw_class=       BRW,
            rw_lifetime=    process_lifetime,
            multiproc=      multiproc,
            verb=           1)

        tasks = [{
            'id':               id,
            'sec':              random.randrange(1, max_sec),
            'exception_prob':   exception_prob}
            for id in range(n_tasks)]

        results = ompr.process(tasks, exit=False)
        print(f'({len(results)}) {results}')
        self.assertTrue(isinstance(results[0], str))
        self.assertEqual(n_tasks, len(results))
        self.assertEqual(len(tasks), len(results))

        # additional 30 tasks
        tasks = [{
            'id':   id,
            'sec':  random.randrange(1, max_sec)}
            for id in range(30)]

        results = ompr.process(tasks, exit=False)
        print(f'({len(results)}) {results}')
        self.assertTrue(isinstance(results[0], str))
        self.assertEqual(len(tasks), len(results))

        ompr.exit()


if __name__ == '__main__':
    unittest.main()