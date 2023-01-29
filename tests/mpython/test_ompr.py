import random
import time
import unittest
from typing import Union

from pypaq.mpython.ompr import OMPRunner, RunningWorker, OMPRException


# basic RunningWorker with random exception
class BRW(RunningWorker):
    def process(
            self,
            id: int,
            sec: Union[float,int],
            exception_prob: float=  0.0) -> object:
        time.sleep(sec)
        if random.random() < exception_prob: raise Exception('randomly crashed')
        return f'{id}_{sec}'


class TestOMPR(unittest.TestCase):

    def test_OMPR_base(self):

        n_tasks =   100
        workers =   10
        min_time =  0.5
        max_time =  1.7

        ompr = OMPRunner(
            rw_class=       BRW,
            devices=        [None] * workers,
            report_delay=   2,
            #loglevel=      10,
        )
        tasks = [{
            'id':   id,
            'sec':  min_time + random.random() * (max_time-min_time)}
            for id in range(n_tasks)]

        ompr.process(tasks)
        results = ompr.get_all_results()

        print(f'({len(results)}) {results}')

        self.assertTrue(isinstance(results[0], str))
        self.assertEqual(len(tasks), len(results))

        results = ompr.get_all_results()
        self.assertTrue(results == [])

        ompr.exit()

    # results received one by one
    def test_OMPR_one_by_one(self):

        n_tasks =   50
        workers =   10
        min_time =  0.5
        max_time =  1.7

        ompr = OMPRunner(
            rw_class=       BRW,
            devices=        [None] * workers,
            report_delay=   2,
            #loglevel=       10,
        )
        tasks = [{
            'id':   id,
            'sec':  min_time + random.random() * (max_time-min_time)}
            for id in range(n_tasks)]

        ompr.process(tasks)
        results = []
        while len(results) < n_tasks:
            print(f'got {len(results)} results')
            results.append(ompr.get_result())

        print(f'({len(results)}) {results}')

        self.assertTrue(isinstance(results[0], str))
        self.assertEqual(len(tasks), len(results))

        # check sorted
        prev = -1
        for res in results:
            curr = int(res.split('_')[0])
            self.assertTrue(curr > prev)
            prev = curr

        result = ompr.get_result(block=False)
        self.assertTrue(result is None)

        ompr.exit()

    # not sorted results
    def test_OMPR_one_by_one_not_sorted(self):

        n_tasks =   50
        workers =   10
        min_time =  0.5
        max_time =  1.7

        ompr = OMPRunner(
            rw_class=           BRW,
            devices=            [None] * workers,
            ordered_results=    False,
            report_delay=       2,
            #loglevel=           10,
        )
        tasks = [{
            'id':   id,
            'sec':  min_time + random.random() * (max_time-min_time)}
            for id in range(n_tasks)]

        ompr.process(tasks)
        results = []
        while len(results) < n_tasks:
            print(f'got {len(results)} results')
            results.append(ompr.get_result())

        print(f'({len(results)}) {results}')

        self.assertTrue(isinstance(results[0], str))
        self.assertEqual(len(tasks), len(results))

        result = ompr.get_result(block=False)
        self.assertTrue(result is None)

        ompr.exit()

    # process lifetime
    def test_OMPR_lifetime(self):

        n_tasks =           100
        workers =           10
        min_time =          0.5
        max_time =          1.7
        process_lifetime =  2

        ompr = OMPRunner(
            rw_class=       BRW,
            rw_lifetime=    process_lifetime,
            devices=        [None] * workers,
            report_delay=   2,
            #loglevel=       10,
        )

        tasks = [{
            'id':   id,
            'sec':  min_time + random.random() * (max_time-min_time)}
            for id in range(n_tasks)]

        print(f'tasks: ({len(tasks)}) {tasks}')
        ompr.process(tasks)
        results = ompr.get_all_results()
        print(f'results: ({len(results)}) {results}')
        self.assertTrue(isinstance(results[0], str))
        self.assertEqual(len(tasks), len(results))

        # additional 30 tasks
        tasks = [{
            'id':   id,
            'sec':  min_time + random.random() * (max_time-min_time)}
            for id in range(30)]

        print(f'tasks: ({len(tasks)}) {tasks}')
        ompr.process(tasks)
        results = ompr.get_all_results()
        print(f'results: ({len(results)}) {results}')
        self.assertTrue(isinstance(results[0], str))
        self.assertEqual(len(tasks), len(results))

        ompr.exit()

        # OMPRunner example with process lifetime and exceptions

    # exceptions
    def test_OMPR_exceptions(self):

        n_tasks =       100
        workers =       10
        min_time =      0.5
        max_time =      1.7
        exception_prob= 0.3

        ompr = OMPRunner(
            rw_class=       BRW,
            devices=        [None] * workers,
            report_delay=   2,
            #loglevel=       10,
        )

        tasks = [{
            'id':               id,
            'sec':              min_time + random.random() * (max_time-min_time),
            'exception_prob':   exception_prob}
            for id in range(n_tasks)]

        print(f'tasks: ({len(tasks)}) {tasks}')
        ompr.process(tasks)
        results = ompr.get_all_results()
        print(f'results: ({len(results)}) {results}')
        for r in results:
            self.assertTrue(isinstance(r,str) or isinstance(r,OMPRException))
        self.assertEqual(len(tasks), len(results))

        # additional 30 tasks
        tasks = [{
            'id':   id,
            'sec':  min_time + random.random() * (max_time-min_time)}
            for id in range(30)]

        print(f'tasks: ({len(tasks)}) {tasks}')
        ompr.process(tasks)
        results = ompr.get_all_results()
        print(f'results: ({len(results)}) {results}')
        for r in results:
            self.assertTrue(isinstance(r,str) or isinstance(r,OMPRException))
        self.assertEqual(len(tasks), len(results))

        ompr.exit()

    # task timeout
    def test_OMPR_timeout(self):

        n_tasks =       100
        workers =       10
        min_time =      0.5
        max_time =      1.7
        task_timeout =  1

        ompr = OMPRunner(
            rw_class=       BRW,
            devices=        [None] * workers,
            task_timeout=   task_timeout,
            report_delay=   2,
            #loglevel=       10,
        )

        tasks = [{
            'id':   id,
            'sec':  min_time + random.random() * (max_time-min_time)}
            for id in range(n_tasks)]

        print(f'tasks: ({len(tasks)}) {tasks}')
        ompr.process(tasks)
        results = ompr.get_all_results()
        print(f'results: ({len(results)}) {results}')
        self.assertEqual(len(tasks), len(results))

        ompr.exit()

    # lifetime + exceptions + timeout
    def test_OMPR_all_together(self):

        n_tasks =           100
        workers =           10
        min_time =          0.5
        max_time =          1.7
        exception_prob =    0.3
        task_timeout =      1
        process_lifetime =  2

        ompr = OMPRunner(
            rw_class=       BRW,
            rw_lifetime=    process_lifetime,
            devices=        [None] * workers,
            task_timeout=   task_timeout,
            report_delay=   2,
            #loglevel=       10,
        )

        tasks = [{
            'id':               id,
            'sec':              min_time + random.random() * (max_time-min_time),
            'exception_prob':   exception_prob}
            for id in range(n_tasks)]

        print(f'tasks: ({len(tasks)}) {tasks}')
        ompr.process(tasks)
        results = ompr.get_all_results()
        print(f'results: ({len(results)}) {results}')
        self.assertEqual(len(tasks), len(results))

        ompr.exit()

    # many timeouts
    def test_OMPR_many_timeouts(self):

        n_tasks =           10000
        min_time =          0.9
        max_time =          1.5
        timeout =           1
        exception_prob =    0.5

        ompr = OMPRunner(
            rw_class=           BRW,
            devices=            'all',
            task_timeout=       timeout,
            log_RWW_exception=  False,
            report_delay=       2,
            #loglevel=           10,
        )

        tasks = [{
            'id':               id,
            'sec':              min_time + random.random() * (max_time-min_time),
            'exception_prob':   exception_prob,
        }
            for id in range(n_tasks)]

        print(f'tasks: ({len(tasks)})')
        ompr.process(tasks)
        results = ompr.get_all_results()
        print(f'results: ({len(results)})')
        self.assertEqual(len(tasks), len(results))

        ompr.exit()

        # OMPRunner with many task timeout

    # many fast tasks
    def test_OMPR_speed(self):

        # Fast RunningWorker
        class FRW(RunningWorker):
            def process(self, id: int) -> int:
                return id

        n_tasks =           100000

        ompr = OMPRunner(
            rw_class=           FRW,
            devices=            'all',
            report_delay=       2,
            #loglevel=           10,
        )

        tasks = [{'id':id} for id in range(n_tasks)]

        print(f'tasks: ({len(tasks)})')
        ompr.process(tasks)
        results = ompr.get_all_results()
        print(f'results: ({len(results)})')
        self.assertEqual(len(tasks), len(results))

        ompr.exit()

    # many timeouts + exceptions
    def test_OMPR_stress(self):

        n_tasks =           10000
        min_time =          0.001
        max_time =          2.2
        timeout =           2
        exception_prob =    0.05

        # Stress RunningWorker with random exception
        class SRW(RunningWorker):
            def process(
                    self,
                    s: list,
                    max_time: float,
                    exception_prob: float=  0.0) -> str:

                if random.random() < exception_prob: raise Exception('randomly crashed')

                s_time = time.time()
                while time.time() - s_time < max_time:
                    random.shuffle(s)
                    s = sorted(s)
                return s

        ompr = OMPRunner(
            rw_class=           SRW,
            devices=            0.7,
            task_timeout=       timeout,
            log_RWW_exception=  False,
            report_delay=       2,
            #loglevel=           10,
        )

        tasks = [{
            's':                list('abcdefghijklmnoprstuvwxyz1234567890'),
            'max_time':         min_time + random.random() * (max_time-min_time),
            'exception_prob':   exception_prob,
        } for _ in range(n_tasks)]

        print(f'got {len(tasks)} tasks')
        ompr.process(tasks)
        results = ompr.get_all_results()
        print(f'got {len(results)} results')
        self.assertEqual(len(tasks), len(results))

        ompr.exit()