"""

 2021 (c) piteren

    OMPRunnerNB - Object based Multi Processing with non-blocking interface

    Tasks may be given as dicts one after another or in packages (list of dicts).
    By default results will be ordered in tasks order.
    Results may be received with two get methods.
    OMPRunnerNB needs to be manually closed with exit().

"""

from collections import deque
import os
import time
from typing import Any, List, Dict, Optional, Tuple
from queue import Empty

from pypaq.lipytools.decorators import timing
from pypaq.mpython.mptools import MultiprocParam, DevicesParam, QMessage, ExSubprocess, Que
from pypaq.mpython.omp import RunningWorker, OMPRunner

# non-blocking OMP, uses qued OMP_IP (internal process) for non-blocking process() interface
class OMPRunnerNB:

    # OMPRunnerNB Internal Process
    class OMP_IP(OMPRunner, ExSubprocess):

        POISON_MSG = {'type':'poison', 'data':None}

        def __init__(
                self,
                ique: Que,                  # tasks and other
                oque: Que,                  # results and exceptions
                ordered_results: bool,
                name=   'OMP_NB_InternalProcess',
                **kwargs):

            self.ordered_results = ordered_results

            # inits as a OMPRunner
            OMPRunner.__init__(
                self,
                name=           name,
                start_allRW=    False,  # those will be started within a subprocess loop
                **kwargs)

            # adds to OMPRunner subprocess properties
            ExSubprocess.__init__(
                self,
                ique=           ique,
                oque=           oque,
                id=             name,
                verb=           self.verb)

        # main loop of OMP_IP, run by ExSubprocess
        def subprocess_method(self):

            if self.verb>0: print(f' > {self.omp_name} (for {self.rw_class.__name__}) loop started')
            self.build_and_start_allRW()

            task_ix = 0                                     # current task index (index of task that will be processed next)
            task_rix = 0                                    # index of task that should be put to self.results_que now
            rww_ntasks = {k: 0 for k in self.rwwD.keys()}   # number of tasks processed by each RWW since restart

            iv_time = time.time()                           # interval report time
            iv_n_tasks = 0                                  # number of tasks finished since last interval
            total_n_tasks = 0                               # total number of tasks processed
            speed_cache = []                                # interval speed cache for moving average
            tasks = deque()                                 # que of tasks ready to be processed (received from the self.tasks_que)
            resources = list(self.rwwD.keys())              # list [id] of all available (not busy) resources
            rww_tasks: Dict[int, Tuple[int, Any]] = {}      # rww.id: (task_ix,task)
            resultsD: Dict[int, Any] = {}                   # results dict {task_ix: result(data)}
            while True:

                break_loop = False # flag to break this loop

                # to allow poison to be received while tasks still present
                qd = self.ique.get_if()
                if qd:
                    if qd['type'] == 'poison': break_loop = True
                    if qd['type'] == 'tasks': tasks += qd['data']

                if not tasks and len(resources) == len(self.rwwD):
                    qd = self.ique.get() # wait here for task or poison
                    if qd['type'] == 'poison': break_loop = True
                    if qd['type'] == 'tasks': tasks += qd['data']

                # try to flush the que
                if not self.ique.empty() or self.ique.qsize():
                    if self.verb>1: print(f'ique is empty - {self.ique.empty()}, it has {self.ique.qsize()} objects, trying to flush in the 2nd loop')
                    qd = self.ique.get_if()
                    while qd:
                        if qd:
                            if qd['type'] == 'poison':
                                break_loop = True
                                break
                            if qd['type'] == 'tasks': tasks += qd['data']
                        qd = self.ique.get_if()

                if break_loop:
                    self._kill_allRW() # RW have to be killed here, from the loop, we want to kill them because it is quicker than to wait for them till finish tasks - we do not need their results here
                    break

                if self.verb>2: print(f' >>> free resources: {len(resources)}, task_ix: {task_ix}, tasks: {len(tasks)}, ique.qsize: {self.ique.qsize()}')
                while resources and tasks: # put all needed resources into work

                    rww_id = resources.pop(0) # take first free resource

                    # restart RWW
                    if self.rw_lifetime and rww_ntasks[rww_id] == self.rw_lifetime:
                        if self.verb>1: print(f' >> restarting RWWrap id: {rww_id}...')
                        self._kill_RW(rww_id)
                        self._build_and_start_RW(rww_id)
                        rww_ntasks[rww_id] = 0

                    # get task, prepare and put message for RWWrap
                    task = tasks.popleft()
                    msg: QMessage = {
                        'type': 'task',
                        'data': task}
                    self.rwwD[rww_id]['rww'].ique.put(msg)
                    rww_tasks[rww_id] = (task_ix,task)
                    if self.verb>2: print(f' >>> put task {task_ix} for RWWrap({rww_id})')

                    task_ix += 1

                # flush que (get messages from RWWraps)
                rww_msgL: List[QMessage] = []
                msg = self.que_RW.get() # at least one
                while msg:
                    rww_msgL.append(msg)
                    msg = self.que_RW.get_if()
                #while not self.que_RW.empty(): rww_msgL.append(self.que_RW.get()) # --- other way, tested but above looks better
                if self.verb>2: print(f' >>> received {len(rww_msgL)} messages from RWWraps')

                for msg in rww_msgL:

                    if msg['type'] == 'result':

                        rww_id = msg['data']['rwwid']
                        rww_ntasks[rww_id] += 1
                        rwt = rww_tasks.pop(rww_id)
                        iv_n_tasks += 1
                        total_n_tasks += 1
                        if self.verb>2: print(f' >> got result ix: {rwt[0]}')
                        if self.ordered_results: resultsD[rwt[0]] = msg['data']['result']
                        else: self.oque.put(msg['data']['result'])
                        resources.append(rww_id)

                        # flush resultsD
                        while task_rix in resultsD:
                            self.oque.put(resultsD.pop(task_rix))
                            task_rix += 1

                    # RWWrap exception
                    else:
                        assert 'ex_' in msg['type'], 'ERR: unknown RWWrap message received!'
                        rww_id = msg['data']
                        if self.verb>0 or self.print_exceptions: print(f' > {self.omp_name} received exception message: {msg}, not finished task ix: {rww_tasks[rww_id][0]}; .. recreating RWWrap and putting task again')

                        # close rww
                        rww = self.rwwD[rww_id]['rww']
                        rww.join() # we cannot kill that process since only alive process can be killed
                        rww.close()
                        self.rwwD[rww_id]['rww'] = None

                        # rebuild, put message again
                        self._build_and_start_RW(rww_id)
                        msg: QMessage = {
                            'type': 'task',
                            'data': rww_tasks[rww_id][1]}
                        self.rwwD[rww_id]['rww'].ique.put(msg)

                if time.time()-iv_time > self.report_delay and self.verb>0:
                    iv_speed = iv_n_tasks/((time.time()-iv_time)/60)
                    speed_cache.append(iv_speed)
                    if len(speed_cache) > 5: speed_cache.pop(0)
                    speed = sum(speed_cache) / len(speed_cache)
                    if speed != 0:
                        if speed > 10:      speed_str = f'{int(speed)} tasks/min'
                        else:
                            if speed > 1:   speed_str = f'{speed:.1f} tasks/min'
                            else:           speed_str = f'{1 / speed:.1f} min/task'
                        tasks_left = len(tasks)
                        est = tasks_left / speed
                        progress = total_n_tasks / (total_n_tasks+tasks_left)
                        print(f' > ({progress * 100:4.1f}% {time.strftime("%H:%M:%S")}) speed: {speed_str}, EST:{est:.1f}min')
                    else: print(f' > processing speed unknown yet..')
                    iv_time = time.time()
                    iv_n_tasks = 0
                    if self.verb>1: print(self._get_RW_info())

        def after_exception_handle_run(self):
            self._kill_allRW()
            if self.verb>2: print(f' > {self.omp_name} killed all RW after exception occurred')

        # method to call out of the process (to exit it)
        def exit(self) -> None:

            if self.alive:
                self.ique.put(self.POISON_MSG)

                while self.alive:
                    # flush the oque
                    while True:
                        res = self.oque.get_if()
                        if res is None: break
                    self.join(timeout=0.0001)

    def __init__(
            self,
            rw_class: type(RunningWorker),          # RunningWorker class that will run() given tasks
            rw_init_kwargs: Optional[Dict]= None,   # RunningWorker __init__ kwargs
            rw_lifetime: Optional[int]=     None,   # RunningWorker lifetime, for None or 0 is unlimited, for N <1,n> each RW will be restarted after processing N tasks
            multiproc: MultiprocParam=      'auto', # multiprocessing cores; auto, all, off, 1-N
            devices: DevicesParam=          None,   # alternatively may be set (None gives priority to multiproc), if given then RW must accept 'devices' param
            name=                           'OMPRunnerNB',
            ordered_results=                True,   # returns results in the order of tasks
            report_delay: int or str=       'auto', # num sec between speed_report
            print_exceptions=               True,   # allows to pint exceptions even for verb==0
            raise_RWW_exception=            False,  # forces RWW to raise exceptions (.. all but KeyboardInterrupt)
            verb=                           0):

        self.verb = verb
        self.omp_name = name
        if report_delay == 'auto': report_delay = 10 if self.verb>1 else 30

        if self.verb>0: print(f'\n*** {self.omp_name} (pid: {os.getpid()}) inits for {rw_class.__name__}')

        self._tasks_que = Que()         # que of tasks to be processed
        self._results_que = Que()       # ready results que
        self._n_tasks_received = 0      # number of tasks received from user till now
        self._n_results_returned = 0    # number of results returned to user till now

        self._internal_processor = OMPRunnerNB.OMP_IP(
            rw_class=               rw_class,
            rw_init_kwargs=         rw_init_kwargs if rw_init_kwargs else {},
            rw_lifetime=            rw_lifetime,
            multiproc=              multiproc,
            devices=                devices,
            ique=                   self._tasks_que,
            oque=                   self._results_que,
            ordered_results=        ordered_results,
            report_delay=           report_delay,
            print_exceptions=       print_exceptions,
            raise_RWW_exception=    raise_RWW_exception,
            verb=                   self.verb)
        self._internal_processor.start()

    # non-blocking method that starts processing given tasks
    def process(self, tasks: dict or List[dict]):
        if type(tasks) is dict: tasks = [tasks]
        self._tasks_que.put({'type':'tasks','data':tasks})
        self._n_tasks_received += len(tasks)

    # returns single result
    def get_result(
            self,
            block=  True):  # blocks execution till result ready and returned, for False does not block and returns None
        if self._n_results_returned == self._n_tasks_received and not block:
            if self.verb>0: print(f'OMPRunner get_result() returns None since n_results_returned == n_tasks_received (does not have any ready result) and is not blocking')
            return None
        else:
            if block:   result = self._results_que.get()
            else:       result = self._results_que.get_if()
            if result: self._n_results_returned += 1
            return result

    # returns results of all tasks put up to NOW
    def get_all_results(self):
        results = []
        if self._n_results_returned == self._n_tasks_received:
            if self.verb>0: print(f'OMPRunner get_all_results() returns [] since n_results_returned == n_tasks_received')
        else:
            n_results = self._n_tasks_received - self._n_results_returned
            while len(results) < n_results: results.append(self._results_que.get())
        self._n_results_returned += len(results)
        return results

    def exit(self):
        if self._n_results_returned != self._n_tasks_received: print(f'WARNING: {self.omp_name} exits while not all results were returned to user!')
        self._internal_processor.exit()
        if self.verb>0: print(f' > {self.omp_name}: internal processor stopped, {self.omp_name} exits.')


# basic OMPRunnerNB example
@timing
def example_basic_OMPRunnerNB(
        multiproc: MultiprocParam=  10,
        n_tasks: int=               50,
        max_sec: int=               5):

    import random

    # basic RunningWorker
    class BRW(RunningWorker):
        def process(self,
                    id: int,
                    sec: int) -> object:
            time.sleep(sec)
            return f'{id}_{sec}'

    ompr = OMPRunnerNB(
        rw_class=       BRW,
        multiproc=      multiproc,
        verb=           1)
    tasks = [{'id': id, 'sec': random.randrange(1, max_sec)} for id in range(n_tasks)]
    ompr.process(tasks)
    results = ompr.get_all_results()
    ompr.exit()

    print(f'({len(results)}) {results}')

# OMPRunnerNB example with process lifetime and exceptions
@timing
def example_more_OMPRunnerNB(
        multiproc: MultiprocParam=  10,
        n_tasks: int=               100,
        max_sec: int=               5,
        process_lifetime=           2,
        exception_prob=             0.1):

    import random
    import time

    # basic RunningWorker
    class BRW(RunningWorker):
        def process(self,
                    id: int,
                    sec: int) -> object:
            if random.random() < exception_prob: raise Exception('randomly crashed')
            time.sleep(sec)
            return f'{id}_{sec}'

    ompr = OMPRunnerNB(
        rw_class=       BRW,
        rw_lifetime=    process_lifetime,
        multiproc=      multiproc,
        verb=           2)

    tasks = [{'id': id, 'sec': random.randrange(1, max_sec)} for id in range(n_tasks)]
    ompr.process(tasks)
    results = ompr.get_all_results()
    print(f'({len(results)}) {results}')

    tasks = [{'id': id, 'sec': random.randrange(1, max_sec)} for id in range(30)] # additional 30 tasks
    ompr.process(tasks)
    results = ompr.get_all_results()
    print(f'({len(results)}) {results}')

    ompr.exit()

if __name__ == '__main__':
    example_basic_OMPRunnerNB()
    #example_more_OMPRunnerNB()