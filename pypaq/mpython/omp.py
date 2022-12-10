"""

 2021 (c) piteren

    OMPRunner - Object based Multi Processing with non-blocking interface

    OMPRunner processes given tasks using RunningWorker class objects.
        RunningWorker must be inherited and its process() implemented, it takes task via **kwargs and returns task result.
        Result returned by RunningWorker may be Any.

    Tasks may be given for OMPRunner as dicts: one dict after another or in packages (List[dict]).
    Results may be received with two get methods (single or all) and by default will be ordered with tasks order.
    OMPRunner needs to be manually closed with exit().

    There are two main policies of RunningWorker lifecycle:
        1st - RunningWorker is closed after processing some task (1 typically but may be N)
        2nd - RunningWorker is closed only when crashes or with the OMP exit
        Each policy has job specific pros and cons. By default second is activated with 'rw_lifetime=None'.
        Pros of 'rw_lifetime=None' are that all RunningWorkers are initialized once while OMP inits. It saves a time.
        Cons are that some data may be shared between tasks - it may be a problem with some libraries. Also memory
        kept by the RunningWorker ma grow with the time (while processing many tasks).

"""

from abc import ABC, abstractmethod
from collections import deque
from inspect import getfullargspec
import os
import psutil
import time
from typing import Any, List, Dict, Optional, Tuple

from pypaq.mpython.devices import DevicesParam, get_devices
from pypaq.mpython.mptools import QMessage, ExSubprocess, Que


# interface of running object - processes task given with kwargs and returns result
class RunningWorker(ABC):

    @ abstractmethod # processing method to be implemented
    def process(self, **kwargs) -> Any: pass


class OMPRunner:

    # Object Multi Processing Internal Process
    class OMP_IP(ExSubprocess):

        POISON_MSG = QMessage(type='poison', data=None)

        # wraps RunningWorker with Exception Managed Subprocess
        class RWWrap(ExSubprocess):

            def __init__(
                    self,
                    rw_class: type(RunningWorker),
                    rw_init_kwargs: Dict,
                    raise_unk_exception=    False,
                    **kwargs):

                ExSubprocess.__init__(
                    self,
                    raise_unk_exception=    raise_unk_exception,
                    **kwargs)

                self.rw_class = rw_class
                self.rw_init_kwargs = rw_init_kwargs

                if self.verb>0: print(f' > RWWrap({self.id}) initialized')

            # loop for processing RW tasks, this method will be run within subprocess (ExSubprocess)
            def subprocess_method(self):

                if self.verb>0: print(f' > RWWrap ({self.id}) pid: {os.getpid()} inits RunningWorker')
                rwo = self.rw_class(**self.rw_init_kwargs)
                if self.verb>0: print(f' > RWWrap ({self.id}) starts process loop..')

                while True:
                    ompr_msg: QMessage = self.ique.get()
                    if ompr_msg.type == 'break': break
                    if ompr_msg.type == 'hold_check': self.oque.put(QMessage(type='hold_ready', data=None))
                    if ompr_msg.type == 'task':
                        result = rwo.process(**ompr_msg.data)
                        res_msg = QMessage(
                            type=   'result',
                            data=   {
                                'rwwid':    self.id,
                                'result':   result})
                        self.oque.put(res_msg)

                if self.verb>0: print(f' > RWWrap (id: {self.id}) finished process loop')


        def __init__(
                self,
                name: str,
                ique: Que,                              # tasks and other messages
                oque: Que,                              # results and exception_results
                rw_class: type(RunningWorker),
                rw_init_kwargs: Optional[Dict],
                rw_lifetime: Optional[int],
                devices: DevicesParam,
                ordered_results: bool,
                report_delay: int or str,
                restart_ex_tasks,
                print_exceptions,
                raise_RWW_exception,
                verb):

            self.omp_name = name
            if verb>0: print(f'\n*** {self.omp_name} (pid: {os.getpid()}) inits for {rw_class.__name__}')

            # adds to OMPRunner Exception Managed Subprocess properties
            ExSubprocess.__init__(
                self,
                ique=           ique,
                oque=           oque,
                id=             self.omp_name,
                verb=           verb)

            self.que_RW = Que() # here OMP receives messages from RW ('result' or 'ex_..'/exception)

            self.rw_class = rw_class
            self.rw_lifetime = rw_lifetime

            if report_delay == 'auto': report_delay = 10 if self.verb > 1 else 30
            self.report_delay = report_delay

            self.ordered_results = ordered_results

            self.restart_ex_tasks = restart_ex_tasks
            self.print_exceptions = print_exceptions
            self.raise_RWW_exception = raise_RWW_exception

            pms = getfullargspec(self.rw_class).args
            assert 'device' not in pms, 'ERR: please rename "device" into "devices" in params of RunningWorker'
            dev_param_name = 'devices' if 'devices' in pms else None

            devices = get_devices(devices=devices, namespace=None)

            # prepare RunningWorkers arguments dictionary
            if not rw_init_kwargs: rw_init_kwargs = {}
            self.rwwD: Dict[int, Dict] = {}  # {rww.id: {'rw_init_kwargs':{}, 'rww':RWWrap}}
            for id, dev in enumerate(devices):
                kwD = {}
                kwD.update(rw_init_kwargs)
                if dev_param_name: kwD[dev_param_name] = dev
                self.rwwD[id] = {
                    'rw_init_kwargs':   kwD,
                    'rww':              None}


        # builds and starts RWWrap
        def _build_and_start_RW(self, id:int):
            assert self.rwwD[id]['rww'] is None
            self.rwwD[id]['rww'] = OMPRunner.OMP_IP.RWWrap(
                ique=                   Que(),
                oque=                   self.que_RW,
                id=                     id,
                rw_class=               self.rw_class,
                rw_init_kwargs=         self.rwwD[id]['rw_init_kwargs'],
                raise_unk_exception=    self.raise_RWW_exception,
                verb=                   self.verb-1)
            self.rwwD[id]['rww'].start()

        def build_and_start_allRW(self):
            n_started = 0
            for id in self.rwwD:
                if self.rwwD[id]['rww'] is None:
                    self._build_and_start_RW(id)
                    n_started += 1
            if self.verb>0 and n_started: print(f' > {self.omp_name} (for {self.rw_class.__name__}) built and started {n_started} RunningWorkers')

        # kills RWWrap
        def _kill_RW(self, id:int):
            self.rwwD[id]['rww'].kill()
            while True: # we have to flush the RW ique
                ind = self.rwwD[id]['rww'].ique.get_if()
                #if ind: print(f'@@@ got ind of RW {id}: {ind}')
                if not ind: break
            self.rwwD[id]['rww'].join()
            self.rwwD[id]['rww'] = None
            if self.verb>1: print(f' >> killed and joined RWWrap id: {id}..')


        def _kill_allRW(self):
            for id in self.rwwD:
                if self.rwwD[id]['rww'] is not None:
                    self._kill_RW(id)

        # this method checks if all RunningWorkers are ready to process tasks
        def hold_till_RW_ready(self):
            if self.verb>0: print(f' > hold: {self.omp_name} is checking RW readiness..')
            for id in self.rwwD:
                if self.rwwD[id]['rww'] is None:
                    print(f'WARNING: some RW are not started, cannot hold!!!')
                    return
            for id in self.rwwD: self.rwwD[id]['rww'].ique.put(QMessage(type='hold_check', data=None))
            for _ in self.rwwD: self.que_RW.get()
            if self.verb>0: print(' > hold: all RW are ready')

        # returns string with information about subprocesses
        def _get_RW_info(self) -> str:

            omp_id = os.getpid()
            omp_mem = int(psutil.Process(omp_id).memory_info().rss / 1024 ** 2)
            vm = psutil.virtual_memory()
            used = vm.used / 1024 ** 3

            num_all = len(self.rwwD)
            num_alive = sum([1 for rww_id in self.rwwD if self.rwwD[rww_id]['rww'] is not None and self.rwwD[rww_id]['rww'].alive])
            num_closed = sum([1 for rww_id in self.rwwD if self.rwwD[rww_id]['rww'] is not None and self.rwwD[rww_id]['rww'].closed])
            alive_info = f'{num_all}= alive:{num_alive} closed:{num_closed}'

            rww_mem = [int(psutil.Process(self.rwwD[id]['rww'].pid).memory_info().rss / 1024 ** 2) for id in self.rwwD if self.rwwD[id]['rww'].alive]
            rww_mem.sort(reverse=True)

            tot_mem = omp_mem + sum(rww_mem)
            s = f'  # OMPRunner mem: {omp_mem}MB, omp+sp/used: {tot_mem/1024:.1f}/{used:.1f}GB ({int(vm.percent)}%VM)\n'
            if len(rww_mem) > 6: s += f'  # subproc: {rww_mem[:3]}-{int(sum(rww_mem)/len(rww_mem))}-{rww_mem[-3:]} ({alive_info})'
            else:                s += f'  # subproc: {rww_mem} ({alive_info})'
            return s

        # main loop of OMP_IP, run by ExSubprocess
        def subprocess_method(self):

            if self.verb>0: print(f' > {self.omp_name} (for {self.rw_class.__name__}) loop started with {len(self.rwwD)} subprocesses')
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
            rww_tasks: Dict[int, Tuple[int, Any]] = {}      # {rww.id: (task_ix,task)}
            resultsD: Dict[int, Any] = {}                   # results dict {task_ix: result(data)}
            while True:

                break_loop = False # flag to break this loop

                # to allow poison to be received while tasks still present
                msg = self.ique.get_if()
                if msg:
                    if msg.type == 'poison': break_loop = True
                    if msg.type == 'tasks': tasks += msg.data

                if not tasks and len(resources) == len(self.rwwD):
                    msg = self.ique.get() # wait here for task or poison
                    if msg.type == 'poison': break_loop = True
                    if msg.type == 'tasks': tasks += msg.data

                # try to flush the que
                if not self.ique.empty() or self.ique.qsize():
                    if self.verb>1: print(f'ique is empty - {self.ique.empty()}, it has {self.ique.qsize()} objects, trying to flush in the 2nd loop')
                    while True:
                        msg = self.ique.get_if()
                        if msg:
                            if msg.type == 'poison':
                                break_loop = True
                                break
                            if msg.type == 'tasks': tasks += msg.data
                        else: break

                if break_loop:
                    self._kill_allRW() # RW have to be killed here, from the loop, we want to kill them because it is quicker than to wait for them till finish tasks - we do not need their results anymore
                    break

                if self.verb>2: print(f' >>> free resources: {len(resources)}, task_ix: {task_ix}, tasks: {len(tasks)}, ique.qsize: {self.ique.qsize()}')
                while resources and tasks: # put all needed resources into work

                    rww_id = resources.pop(0) # take first free resource

                    # restart RWW
                    if self.rw_lifetime and rww_ntasks[rww_id] >= self.rw_lifetime:
                        if self.verb>1: print(f' >> restarting RWWrap id: {rww_id}...')
                        self._kill_RW(rww_id)
                        self._build_and_start_RW(rww_id)
                        rww_ntasks[rww_id] = 0

                    # get task, prepare and put message for RWWrap
                    task = tasks.popleft()
                    msg = QMessage(type='task', data=task)
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

                    if msg.type == 'result':

                        rww_id = msg.data['rwwid']
                        rww_ntasks[rww_id] += 1

                        tix, _ = rww_tasks.pop(rww_id)
                        iv_n_tasks += 1
                        total_n_tasks += 1
                        if self.verb>2: print(f' >> got result of task_ix: {tix}')

                        res_msg = QMessage(type='result', data=msg.data['result'])
                        if self.ordered_results: resultsD[tix] = res_msg
                        else: self.oque.put(res_msg)

                        resources.append(rww_id)

                    # RWWrap exception
                    else:
                        assert 'ex_' in msg.type, 'ERR: unknown RWWrap message received!'

                        rww_id = msg.data
                        tix, tsk = rww_tasks.pop(rww_id)
                        if self.verb>0 or self.print_exceptions:
                            print(f' > {self.omp_name} received exception message: {msg}, not finished task ix: {tix}')
                            print(f' >> recreating RWWrap..')

                        # close rww
                        rww = self.rwwD[rww_id]['rww']
                        rww.join() # we cannot kill that process since only alive process can be killed
                        rww.close()
                        self.rwwD[rww_id]['rww'] = None

                        # rebuild
                        self._build_and_start_RW(rww_id)
                        rww_ntasks[rww_id] = 0

                        if self.restart_ex_tasks:
                            if self.verb>0 or self.print_exceptions: print(f' >> putting task again..')
                            msg = QMessage(type='task', data=tsk)
                            self.rwwD[rww_id]['rww'].ique.put(msg)
                            rww_tasks[rww_id] = (tix,tsk)
                        else:
                            if self.verb>0 or self.print_exceptions: print(f' >> returning exception result..')

                            res_msg = QMessage(type='exception_result', data=f'TASK #{tix} RAISED EXCEPTION')
                            if self.ordered_results: resultsD[tix] = res_msg
                            else: self.oque.put(res_msg)

                            resources.append(rww_id)

                    # flush resultsD
                    while task_rix in resultsD:
                        self.oque.put(resultsD.pop(task_rix))
                        task_rix += 1

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
            devices: DevicesParam=          None,   # alternatively may be set (None gives priority to multiproc), if given then RW must accept 'devices' param
            name=                           'OMPRunner',
            ordered_results=                True,   # returns results in the order of tasks
            report_delay: int or str=       'auto', # num sec between speed_report
            restart_ex_tasks=               True,   # restarts tasks that caused exception, for False returns 'TASK #{tix} RAISED EXCEPTION'
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

        self._internal_processor = OMPRunner.OMP_IP(
            name=                   'OMP_NB_InternalProcess',
            ique=                   self._tasks_que,
            oque=                   self._results_que,
            rw_class=               rw_class,
            rw_init_kwargs=         rw_init_kwargs if rw_init_kwargs else {},
            rw_lifetime=            rw_lifetime,
            devices=                devices,
            ordered_results=        ordered_results,
            report_delay=           report_delay,
            restart_ex_tasks=       restart_ex_tasks,
            print_exceptions=       print_exceptions,
            raise_RWW_exception=    raise_RWW_exception,
            verb=                   self.verb)
        self._internal_processor.start()

    # takes tasks for processing, not blocks and starts processing, does not return anything
    def process(self, tasks: dict or List[dict]):
        if type(tasks) is dict: tasks = [tasks]
        self._tasks_que.put(QMessage(type='tasks', data=tasks))
        self._n_tasks_received += len(tasks)

    # returns single result, may block or not
    def get_result(self, block=True) -> Optional[Any]:
        if self._n_results_returned == self._n_tasks_received:
            if self.verb>0: print(f'OMPRunner get_result() returns None since already returned all results (for all given tasks: n_results_returned == n_tasks_received)')
            return None
        else:
            if block:   msg = self._results_que.get()
            else:       msg = self._results_que.get_if()
            if msg:
                self._n_results_returned += 1
                return msg.data
            return None

    # returns results of all tasks put up to NOW
    def get_all_results(self) -> List[Any]:
        results = []
        n_results = self._n_tasks_received - self._n_results_returned
        while len(results) < n_results: results.append(self.get_result(block=True))
        return results

    def exit(self):
        if self._n_results_returned != self._n_tasks_received: print(f'WARNING: {self.omp_name} exits while not all results were returned to user!')
        self._internal_processor.exit()
        if self.verb>0: print(f' > {self.omp_name}: internal processor stopped, {self.omp_name} exits.')
