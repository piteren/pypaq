"""

 2021 (c) piteren

    OMPRunner - Object based Multi Processing
    OMPRunner processes given tasks using RunningWorker class objects.
        RunningWorker process() must be implemented, it takes task via arguments and returns task result.
        Result returned by RunningWorker may be Any but cannot be None.

    There are two main policies of RunningWorker lifecycle:
        1st - RunningWorker is closed after processing some task (1 typically but may be N)
        2nd - RunningWorker is closed only when crashes or with the OMP exit
        Each policy has job specific pros and cons. By default second is activated with 'rw_lifetime=None'.
        Pros of 'rw_lifetime=None' are that all RunningWorkers are initialized once while OMP inits. It saves a time.
        Cons are that some data may be shared between tasks - it may be a problem with some libraries. Also memory
        kept by the RunningWorker ma grow with the time (while processing many tasks).

    Tasks may be given packages (list of dicts). process() processes tasks and returns results and till then blocks further execution.
    Results will be ordered in tasks order.

"""

from abc import ABC, abstractmethod
from collections import deque
from inspect import getfullargspec
import os
import psutil
import time
from typing import Any, List, Dict, Optional, Tuple

from pypaq.mpython.mptools import sys_res_nfo, MultiprocParam, DevicesParam, QMessage, ExSubprocess, Que
from pypaq.neuralmess.dev_manager import tf_devices


# interface of running object - processes task given with kwargs and returns result
class RunningWorker(ABC):
    @ abstractmethod # processing method to be implemented
    def process(self, **kwargs) -> Any: pass

# basic OMP with blocking process() interface, runs tasks with RunningWorkers objects
class OMPRunner:

    # wraps RunningWorker with Exception Managed Subprocess
    class RWWrap(ExSubprocess):

        def __init__(
                self,
                rw_class: type(RunningWorker),
                rw_init_kwargs: dict,
                raise_unk_exception=    False,
                **kwargs):

            ExSubprocess.__init__(
                self,
                raise_unk_exception=    raise_unk_exception,
                **kwargs)

            self.rw_class = rw_class
            self.rw_init_kwargs = rw_init_kwargs

            if self.verb>0: print(f' > RWWrap({self.id}) initialized')

        # .. loop, this method will be run within subprocess (ExSubprocess)
        def subprocess_method(self):
            if self.verb>0: print(f' > RWWrap ({self.id}) pid: {os.getpid()} inits RunningWorker')
            rwo = self.rw_class(**self.rw_init_kwargs)
            if self.verb>0: print(f' > RWWrap ({self.id}) starts process loop..')

            while True:
                ompr_msg: QMessage = self.ique.get()
                if ompr_msg['type'] == 'break': break
                if ompr_msg['type'] == 'hold_check': self.oque.put({'type':'hold_ready','data':None})
                if ompr_msg['type'] == 'task':
                    result = rwo.process(**ompr_msg['data'])
                    res_msg: QMessage = {
                        'type': 'result',
                        'data': {
                            'rwwid':    self.id,
                            'result':   result}}
                    self.oque.put(res_msg)

            if self.verb>0: print(f' > RWWrap (id: {self.id}) finished process loop')

    def __init__(
            self,
            rw_class: type(RunningWorker),                  # RunningWorker class that will run() given tasks
            rw_init_kwargs: Optional[Dict]= None,           # RunningWorker __init__ kwargs
            rw_lifetime: Optional[int]=     None,           # RunningWorker lifetime, for None or 0 is unlimited, for N <1,n> each RW will be restarted after processing N tasks
            multiproc: MultiprocParam =     'auto',         # multiprocessing cores; auto, all, off, 1-N
            devices: DevicesParam=          None,           # alternatively may be set (None gives priority to multiproc), if given then RW must accept 'devices' param
            name=                           'OMPRunner',
            report_delay: int or str=       'auto',         # num sec between speed_report
            print_exceptions=               True,           # allows to pint exceptions even for verb==0
            raise_RWW_exception=            False,          # forces RWW to raise exceptions (.. all but KeyboardInterrupt)
            start_allRW=                    True,           # allows to not start all RW while init
            verb=                           0):

        self.verb = verb
        self.omp_name = name
        if report_delay == 'auto': report_delay = 10 if self.verb > 1 else 30
        self.report_delay = report_delay
        self.print_exceptions = print_exceptions
        self.raise_RWW_exception = raise_RWW_exception

        if self.verb>0: print(f'\n*** {self.omp_name} (pid: {os.getpid()}) inits for {rw_class.__name__}')

        self.que_RW = Que() # here OMP receives messages from RW (results or exceptions)

        self.rw_class = rw_class
        self.rw_lifetime = rw_lifetime

        # prepare RunningWorkers arguments dictionary
        if not rw_init_kwargs: rw_init_kwargs = {}
        self.rwwD: Dict[int, Dict] = {}  # {rww.id: {'rw_init_kwargs':{}, 'rww':RWWrap}}
        if devices is not None:
            pms = getfullargspec(self.rw_class).args
            dev_param_name = 'device' if 'device' in pms else 'devices'
            devices = tf_devices(devices=devices, verb=verb)
            for id, dev in enumerate(devices):
                kwD = {}
                kwD.update(rw_init_kwargs)
                kwD[dev_param_name] = dev
                self.rwwD[id] = {
                    'rw_init_kwargs':   kwD,
                    'rww':              None}
        else:
            # get n_rww
            if type(multiproc) is int:
                assert multiproc > 0, 'ERR: cannot continue with 0 subprocesses'
                n_rww = multiproc
            else:
                if multiproc == 'off': n_rww = 1                            # off
                else:                  n_rww = sys_res_nfo()['cpu_count']   # auto / all
            for id in range(n_rww):
                self.rwwD[id] = {
                    'rw_init_kwargs':   rw_init_kwargs,
                    'rww':              None}                               # None means that RW is not started (or has been killed)

        if start_allRW: self.build_and_start_allRW()

    # builds and starts RWWrap
    def _build_and_start_RW(self, id:int):
        assert self.rwwD[id]['rww'] is None
        self.rwwD[id]['rww'] = OMPRunner.RWWrap(
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

    def hold_till_RW_ready(self):
        if self.verb>0: print(f' > hold: {self.omp_name} is checking RW readiness..')
        for id in self.rwwD:
            if self.rwwD[id]['rww'] is None:
                print(f'WARNING: some RW are not started, cannot hold!!!')
                return
        for id in self.rwwD: self.rwwD[id]['rww'].ique.put({'type':'hold_check','data': None})
        for _ in self.rwwD: self.que_RW.get()
        if self.verb>0: print(' > hold: all RW are ready')

    def process(
            self,
            tasks: List[dict],
            exit=   True        # exits when finished
    ) -> List[Any]:

        if self.verb>0: print(f' > {self.omp_name} (for {self.rw_class.__name__}) loop started processing {len(tasks)} tasks with {len(self.rwwD)} subprocesses')

        self.build_and_start_allRW()                    # always start allRW in case there are not started

        task_ix = 0                                     # current task index (index of task that will be processed next)
        rww_ntasks = {k: 0 for k in self.rwwD.keys()}   # number of tasks processed by each RWW since restart

        iv_time = time.time()                           # interval report time
        iv_n_tasks = 0                                  # number of tasks finished since last interval
        total_n_tasks = 0                               # total number of tasks processed
        speed_cache = []                                # interval speed cache for moving average
        tasks = deque(tasks)                            # que of tasks to be processed
        num_tasks = len(tasks)                          # number of tasks to process
        resources = list(self.rwwD.keys())              # list [id] of all available (not busy) resources
        rww_tasks: Dict[int, Tuple[int, Any]] = {}      # rww.id: (task_ix,task)
        resultsD: Dict[int, Any] = {}                   # results dict {task_ix: result(data)}
        while len(resultsD) < num_tasks:

            if self.verb>2: print(f' >>> free resources: {len(resources)}, task_ix: {task_ix}, tasks: {len(tasks)}')
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
            if self.verb>2: print(f' >>> received {len(rww_msgL)} messages from RWWraps')

            for msg in rww_msgL:

                if msg['type'] == 'result':

                    rww_id = msg['data']['rwwid']
                    rww_ntasks[rww_id] += 1
                    rwt = rww_tasks.pop(rww_id)
                    iv_n_tasks += 1
                    total_n_tasks += 1
                    if self.verb>2: print(f' >> got result ix: {rwt[0]}')
                    resultsD[rwt[0]] = msg['data']['result']
                    resources.append(rww_id)

                # RWWrap exception
                else:
                    assert 'ex_' in msg['type'], 'ERR: unknown RWWrap message received!'
                    rww_id = msg['data']
                    if self.verb>0 or self.print_exceptions: print(f' > {self.omp_name} received exception message from RW: {msg}, not finished task ix: {rww_tasks[rww_id][0]}; .. recreating RWWrap and putting task again')

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
                    est = (num_tasks - total_n_tasks) / speed
                    progress = total_n_tasks / num_tasks
                    print(f' > ({progress * 100:4.1f}% {time.strftime("%H:%M:%S")}) speed: {speed_str}, EST:{est:.1f}min')
                else: print(f' > processing speed unknown yet..')
                iv_time = time.time()
                iv_n_tasks = 0
                if self.verb>1: print(self._get_RW_info())

        if exit: self.exit()

        return [resultsD[k] for k in range(len(resultsD))]

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

    def exit(self) -> None:
        self._kill_allRW()
        while True: # flush the que_RW
            res = self.que_RW.get_if()
            #if res: print(f'@@@ got res from self.que_RW: {res}')
            if not res: break