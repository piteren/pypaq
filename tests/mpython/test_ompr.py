import random
import time
import unittest

from pypaq.mpython.ompr import OMPRunner, RunningWorker


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


class TestOMPR_NB(unittest.TestCase):

    def test_OMPR_base(self):

        n_tasks = 50
        cores =   10
        max_sec = 3

        ompr = OMPRunner(
            rw_class=       BRW,
            devices=        [None]*cores,
            #loglevel=       10
        )
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

    # results received one by one
    def test_OMPR_one_by_one(self):

        n_tasks = 20
        cores =   7
        max_sec = 3

        ompr = OMPRunner(
            rw_class=       BRW,
            devices=        [None] * cores)
        tasks = [{
            'id':   id,
            'sec':  random.randrange(1, max_sec)}
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

        n_tasks = 20
        cores =   7
        max_sec = 3

        ompr = OMPRunner(
            rw_class=           BRW,
            devices=            [None] * cores,
            ordered_results=    False)
        tasks = [{
            'id':   id,
            'sec':  random.randrange(1, max_sec)}
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

    # OMPRunner example with process lifetime and exceptions
    def test_OMPR_lifetime_exceptions(self):

        n_tasks =           50
        cores =             10
        max_sec =           5
        process_lifetime=   2
        exception_prob=     0.1

        ompr = OMPRunner(
            rw_class=       BRW,
            rw_lifetime=    process_lifetime,
            devices=        [None] * cores)

        tasks = [{
            'id':               id,
            'sec':              random.randrange(1, max_sec),
            'exception_prob':   exception_prob}
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
            'sec':  random.randrange(1, max_sec)}
            for id in range(30)]

        print(f'tasks: ({len(tasks)}) {tasks}')
        ompr.process(tasks)
        results = ompr.get_all_results()
        print(f'results: ({len(results)}) {results}')
        self.assertTrue(isinstance(results[0], str))
        self.assertEqual(len(tasks), len(results))

        ompr.exit()

    # OMPRunner with task timeout
    def test_OMPR_timeout(self):

        n_tasks =           100
        cores =             10
        max_sec =           5
        process_lifetime=   2
        exception_prob=     0.1

        ompr = OMPRunner(
            rw_class=           BRW,
            rw_lifetime=        process_lifetime,
            devices=            [None] * cores,
            task_timeout=       4.0,
            #loglevel=           10,
        )

        tasks = [{
            'id':               id,
            'sec':              random.randrange(1, max_sec),
            'exception_prob':   exception_prob}
            for id in range(n_tasks)]

        print(f'tasks: ({len(tasks)}) {tasks}')
        ompr.process(tasks)
        results = ompr.get_all_results()
        print(f'results: ({len(results)}) {results}')
        self.assertEqual(len(tasks), len(results))

        ompr.exit()