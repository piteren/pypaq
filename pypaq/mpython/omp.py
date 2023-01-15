"""

 2021 (c) piteren

    OMPRunner - Object based Multi Processing Runner

        Tasks may be given for OMPRunner with process() method - one dict after another or in packages (List[dict]).
        Results may be received with two get methods (single or all) and by default will be ordered with tasks order.
        Result returned by RunningWorker may be Any.
        OMPRunner needs to be manually closed with exit().

        OMPRunner processes given tasks with InternalProcessor (IP) that guarantees non-blocking interface of OMPRunner.
        Tasks are processed directly by RunningWorker class objects that are managed by IP.

        RunningWorker must be inherited and its process() implemented.
        process() takes task via **kwargs and returns task result.
        There are two main policies of RunningWorker lifecycle:
            1st - RunningWorker is closed after processing some task (1 typically but may be N)
            2nd - RunningWorker is closed only when crashes or with the OMP exit
            Each policy has job specific pros and cons. By default, second is activated with 'rw_lifetime=None'.
            Pros of 'rw_lifetime=None' are:
            + all RunningWorkers are initialized once while OMP inits. It saves a time.
            Cons are:
            - memory kept by the RunningWorker may grow with the time (while processing many tasks).

"""

from abc import ABC, abstractmethod
from collections import deque
from inspect import getfullargspec
import os
import psutil
import time
from typing import Any, List, Dict, Optional, Tuple, Union

from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.lipytools.moving_average import MovAvg
from pypaq.mpython.devices import DevicesParam, get_devices
from pypaq.mpython.mptools import QMessage, ExSubprocess, Que


class OMPRException(Exception):
    pass

# result returned in case of RWW exception
class ResultOMPRException:
    def __init__(self, exception_msg:str, task:dict):
        self.exception_msg = exception_msg
        self.task = task

# interface of running object - processes task given with kwargs and returns result
class RunningWorker(ABC):

    @ abstractmethod # processing method to be implemented
    def process(self, **kwargs) -> Any: pass


class OMPRunner:

    # Internal Processor of OMPRunner
    class InternalProcessor(ExSubprocess):

        POISON_MSG = QMessage(type='poison', data=None)

        # RWW wraps RunningWorker with Exception Managed Subprocess
        class RWWrap(ExSubprocess):

            def __init__(
                    self,
                    ique: Que,
                    oque: Que,
                    id: int,
                    rw_class: type(RunningWorker),
                    rw_init_kwargs: Dict,
                    raise_unk_exception,
                    logger):
                ExSubprocess.__init__(
                    self,
                    ique=                   ique,
                    oque=                   oque,
                    id=                     id,
                    raise_unk_exception=    raise_unk_exception,
                    logger=                 logger)
                self.rw_class = rw_class
                self.rw_init_kwargs = rw_init_kwargs
                self.logger.info(f'> RWWrap({self.id}) initialized')

            # loop for processing RW tasks, this method will be run within subprocess (ExSubprocess)
            def subprocess_method(self):

                self.logger.info(f'> RWWrap ({self.id}) pid: {os.getpid()} inits RunningWorker')
                rwo = self.rw_class(**self.rw_init_kwargs)
                self.logger.info(f'> RWWrap ({self.id}) starts process loop..')

                while True:
                    ompr_msg: QMessage = self.ique.get()
                    if ompr_msg.type == 'break': break
                    if ompr_msg.type == 'hold_check':self.oque.put(QMessage(type='hold_ready', data=None))
                    if ompr_msg.type == 'task':
                        result = rwo.process(**ompr_msg.data['task'])
                        self.oque.put(QMessage(
                            type=   'result',
                            data=   {
                                'rww_id':   self.id,
                                'task_ix':  ompr_msg.data['task_ix'],
                                'task_try': ompr_msg.data['task_try'],
                                'result':   result}))

                self.logger.info(f'> RWWrap (id: {self.id}) finished process loop')


        def __init__(
                self,
                ique: Que,                              # tasks and other messages
                oque: Que,                              # results and exception_results
                rw_class: type(RunningWorker),
                rw_init_kwargs: Optional[Dict],
                rw_lifetime: Optional[int],
                devices: DevicesParam,
                ordered_results: bool,
                task_timeout: Optional[float],
                restart_ex_tasks: bool,
                log_exceptions: bool,
                raise_RWW_exception: bool,
                report_delay: Optional[int],
                logger):

            self.rw_class = rw_class
            self.ip_name = f'InternalProcessor_for_{self.rw_class.__name__}' # INFO: .name conflicts with Process.name

            self.logger = logger
            self.logger.info(f'*** {self.ip_name} *** (pid: {os.getpid()}) inits..')

            # adds to InternalProcessor Exception Managed Subprocess properties
            ExSubprocess.__init__(
                self,
                ique=           ique,
                oque=           oque,
                id=             self.ip_name,
                logger=         self.logger)

            self.que_RW = Que() # here OMP receives messages from RW ('result' or 'ex_..'/exception)

            if not rw_init_kwargs: rw_init_kwargs = {}
            self.rw_lifetime = rw_lifetime

            devices = get_devices(devices=devices, namespace=None)

            dev_param_name = None
            pms = getfullargspec(self.rw_class).args
            if 'devices' in pms: dev_param_name = 'devices'
            if 'device' in pms: dev_param_name = 'device'

            self.ordered_results = ordered_results
            self.task_timeout = task_timeout
            self.restart_ex_tasks = restart_ex_tasks
            self.log_exceptions = log_exceptions
            self.raise_RWW_exception = raise_RWW_exception
            self.report_delay = report_delay

            # prepare RWW dictionary, keeps RWW id, RW init kwargs, RWW object
            self.rwwD: Dict[int, Dict] = {}  # {rww.id: {'rw_init_kwargs':{}, 'rww':RWWrap}}
            for id, dev in enumerate(devices):
                kwD = {}
                kwD.update(rw_init_kwargs)
                if dev_param_name: kwD[dev_param_name] = dev
                self.rwwD[id] = {
                    'rw_init_kwargs':   kwD,
                    'rww':              None}

        # builds and starts single RWWrap
        def _build_and_start_RWW(self, id:int):
            assert self.rwwD[id]['rww'] is None
            self.rwwD[id]['rww'] = OMPRunner.InternalProcessor.RWWrap(
                ique=                   Que(),
                oque=                   self.que_RW,
                id=                     id,
                rw_class=               self.rw_class,
                rw_init_kwargs=         self.rwwD[id]['rw_init_kwargs'],
                raise_unk_exception=    self.raise_RWW_exception,
                logger=                 get_hi_child(self.logger))
            self.rwwD[id]['rww'].start()
            self.logger.debug(f'> {self.ip_name} built and started RWWrap id: {id}..')

        def build_and_start_allRWW(self):
            n_started = 0
            for id in self.rwwD:
                if self.rwwD[id]['rww'] is None:
                    self._build_and_start_RWW(id)
                    n_started += 1
            self.logger.info(f'> {self.ip_name} built and started {n_started} RunningWorkers')

        # kills single RWWrap
        def _kill_RWW(self, id:int):
            self.rwwD[id]['rww'].kill()
            while True: # we have to flush the RW ique
                ind = self.rwwD[id]['rww'].ique.get_if()
                #if ind: print(f'@@@ got ind of RW {id}: {ind}')
                if not ind: break
            self.rwwD[id]['rww'].join()
            self.rwwD[id]['rww'] = None
            self.logger.debug(f'> {self.ip_name} killed and joined RWWrap id: {id}..')

        def _kill_allRWW(self):
            for id in self.rwwD:
                if self.rwwD[id]['rww'] is not None:
                    self._kill_RWW(id)
            self.logger.info(f'> {self.ip_name} killed and joined {len(self.rwwD)} RunningWorkers')

        # this method checks if all RunningWorkers are ready to process tasks
        def hold_till_allRWW_ready(self):
            self.logger.info(f'> hold: {self.ip_name} is checking RW readiness..')
            for id in self.rwwD:
                if self.rwwD[id]['rww'] is None:
                    print(f'WARNING: some RW are not started, cannot hold!!!')
                    return
            for id in self.rwwD: self.rwwD[id]['rww'].ique.put(QMessage(type='hold_check', data=None))
            for _ in self.rwwD: self.que_RW.get()
            self.logger.info(' > hold: all RW are ready')

        # returns string with information about subprocesses
        def _get_RWW_info(self) -> str:

            ip_id = os.getpid()
            ip_mem = int(psutil.Process(ip_id).memory_info().rss / 1024 ** 2)
            vm = psutil.virtual_memory()
            used = vm.used / 1024 ** 3

            num_all = len(self.rwwD)
            num_alive = sum([1 for rww_id in self.rwwD if self.rwwD[rww_id]['rww'] is not None and self.rwwD[rww_id]['rww'].alive])
            num_closed = sum([1 for rww_id in self.rwwD if self.rwwD[rww_id]['rww'] is not None and self.rwwD[rww_id]['rww'].closed])
            alive_info = f'{num_all}= alive:{num_alive} closed:{num_closed}'

            rww_mem = [int(psutil.Process(self.rwwD[id]['rww'].pid).memory_info().rss / 1024 ** 2) for id in self.rwwD if self.rwwD[id]['rww'].alive]
            rww_mem.sort(reverse=True)

            tot_mem = ip_mem + sum(rww_mem)
            s = f'# {self.ip_name} mem: {ip_mem}MB, omp+sp/used: {tot_mem/1024:.1f}/{used:.1f}GB ({int(vm.percent)}%VM)\n'
            if len(rww_mem) > 6: s += f'# subproc: {rww_mem[:3]}-{int(sum(rww_mem)/len(rww_mem))}-{rww_mem[-3:]} ({alive_info})'
            else:                s += f'# subproc: {rww_mem} ({alive_info})'
            return s

        # main loop of OMP_IP, run by ExSubprocess
        def subprocess_method(self):

            self.logger.info(f'> {self.ip_name} loop starts with {len(self.rwwD)} subprocesses (RWW)')
            self.build_and_start_allRWW()

            next_task_ix = 0                                # next task index (index of task that will be processed next)
            task_result_ix = 0                              # index of task result that should be put to self.results_que now
            rww_ntasks = {k: 0 for k in self.rwwD.keys()}   # number of tasks processed by each RWW since restart

            iv_time = time.time()                           # interval report time
            iv_n_tasks = 0                                  # number of tasks finished since last interval
            n_tasks_processed = 0                           # number of tasks processed (total)
            speed_cache = MovAvg(factor=0.2)                # speed moving average
            tasks_que = deque()                             # que of (task_ix, task_try, task) to be processed (received from the self.tasks_que or restarted after exception)
            resources = list(self.rwwD.keys())              # list [rww_id] of all available (not busy) resources
            rww_tasks: Dict[int, Tuple[int,int,Any]] = {}   # {rww.id: (task_ix,task_try,task)}
            task_times: Dict[int, float] = {}               # {rww.id: time} here we note time when RWW got current task, it is a dict but we also use property that order is maintained
            killed_tasks: Dict[int, Tuple[int,int]] = {}    # {rww_id: (task_ix,task_try)} here we note tasks that may come from RWW that were killed, we need to get rid of them

            resultsD: Dict[int, Any] = {}                   # results dict {task_ix: result(data)}
            while True:

                break_loop = False # flag to break this loop

                # to allow poison to be received while tasks still present
                msg = self.ique.get_if()
                if msg:
                    if msg.type == 'poison':
                        break_loop = True
                    if msg.type == 'tasks':
                        for task in msg.data:
                            tasks_que.append((next_task_ix, 0, task))
                            next_task_ix += 1

                if not tasks_que and len(resources) == len(self.rwwD):
                    msg = self.ique.get() # wait here for task or poison
                    if msg.type == 'poison':
                        break_loop = True
                    if msg.type == 'tasks':
                        for task in msg.data:
                            tasks_que.append((next_task_ix, 0, task))
                            next_task_ix += 1

                # try to flush the que
                if not self.ique.empty() or self.ique.qsize():
                    self.logger.debug(f'ique is empty - {self.ique.empty()}, it has {self.ique.qsize()} objects, trying to flush in the 2nd loop')
                    while True:
                        msg = self.ique.get_if()
                        if msg:
                            if msg.type == 'poison':
                                break_loop = True
                                break
                            if msg.type == 'tasks':
                                for task in msg.data:
                                    tasks_que.append((next_task_ix, 0, task))
                                    next_task_ix += 1
                        else: break

                if break_loop:
                    # all RWW have to be killed here, from the loop
                    # we want to kill them because it is quicker than waiting for them till finish tasks
                    # - we do not need their results anymore
                    self._kill_allRWW()
                    break

                self.logger.debug(f'>> free resources: {len(resources)}, tasks_que len: {len(tasks_que)}, ique.qsize: {self.ique.qsize()}')

                # put all needed resources into work
                while resources and tasks_que:

                    rww_id = resources.pop(0) # take first free resource

                    # eventually restart RWW (lifetime limit)
                    if self.rw_lifetime and rww_ntasks[rww_id] >= self.rw_lifetime:
                        self.logger.debug(f'>> restarting RWWrap id: {rww_id}...')
                        self._kill_RWW(rww_id)
                        self._build_and_start_RWW(rww_id)
                        rww_ntasks[rww_id] = 0

                    # get first task, prepare and put message for RWWrap
                    task_ix, task_try, task = tasks_que.popleft()
                    msg = QMessage(
                        type=   'task',
                        data=   {'task_ix':task_ix, 'task_try':task_try, 'task':task})
                    self.rwwD[rww_id]['rww'].ique.put(msg)
                    rww_tasks[rww_id] = (task_ix, task_try, task)
                    task_times[rww_id] = time.time()

                    self.logger.debug(f'>> put task {task_ix} for RWWrap({rww_id})')

                ### flush que_RW (get messages from RWWraps)
                rww_msgL: List[QMessage] = []

                ### get at least one
                # do it in a timeout loop to check for over-timed RWW
                if self.task_timeout:
                    msg = None
                    while not msg:
                        msg = self.que_RW.get_timeout(timeout=self.task_timeout/5)

                        if task_times:
                            oldest_rww_id = list(task_times.keys())[0]
                            if time.time() - task_times[oldest_rww_id] > self.task_timeout:
                                self.rwwD[oldest_rww_id]['rww'].kill()      # kill him
                                task_ix, task_try, _ = rww_tasks[oldest_rww_id] # get his task data
                                killed_tasks[oldest_rww_id] = (task_ix, task_try)
                                if msg: rww_msgL.append(msg)                # do not lose good msg
                                # prepare msg for killed RWW (killed RRW does not return message after kill)
                                msg = QMessage(
                                    type=   f'ex_timeout_killed, ExSubprocess id: {oldest_rww_id}',
                                    data=   oldest_rww_id)  # return ID here to allow process identification

                else: msg = self.que_RW.get()

                # get more (ready)
                while msg:
                    rww_msgL.append(msg)
                    msg = self.que_RW.get_if()
                self.logger.debug(f'>> received {len(rww_msgL)} messages from RWWraps')

                # eventually get rid of task from killed RWW
                # INFO: it is possible that killed process returned its result in the meanwhile (while processing kill)
                # eventually clean_up killed_tasks
                cleaned_rww_msgL = []   # new, clean lost
                clean_rww_id = []       # clean those RWW from killed_tasks
                for msg in rww_msgL:
                    msg_ok = True
                    for rww_id in killed_tasks:
                        if msg.type == 'result' and msg.data['rww_id'] == rww_id:
                            clean_rww_id.append(rww_id)
                            if msg.data['task_ix'] == killed_tasks[rww_id][0] and msg.data['task_try'] == killed_tasks[rww_id][1]:
                                msg_ok = False
                    if msg_ok: cleaned_rww_msgL.append(msg)
                for rww_id in clean_rww_id:
                    killed_tasks.pop(rww_id)
                rww_msgL = cleaned_rww_msgL

                # process all RWW messages
                for msg in rww_msgL:

                    if msg.type == 'result':

                        rww_id = msg.data['rww_id']
                        rww_ntasks[rww_id] += 1

                        task_ix, task_try, _ = rww_tasks.pop(rww_id) # ..could also be get from msg.data
                        task_times.pop(rww_id)

                        iv_n_tasks += 1
                        n_tasks_processed += 1
                        self.logger.debug(f'>> got result of task_ix: {task_ix} try: {task_try}')

                        res_msg = QMessage(type='result', data=msg.data['result'])
                        if self.ordered_results: resultsD[task_ix] = res_msg
                        else: self.oque.put(res_msg)

                        resources.append(rww_id)

                    # RWWrap exception
                    else:
                        if 'ex_' not in msg.type: raise OMPRException('ERR: unknown RWWrap message received!')

                        rww_id = msg.data
                        task_ix, task_try, task = rww_tasks.pop(rww_id)
                        task_times.pop(rww_id)

                        if self.log_exceptions:
                            self.logger.warning(f'> {self.ip_name} received exception message: {msg.type}, not finished task_ix: {task_ix}, recreating RWWrap..')

                        # close rww
                        rww = self.rwwD[rww_id]['rww']
                        if rww is not None: # if RWW was killed it is None then
                            rww.join() # we cannot kill that process since only alive process can be killed
                            rww.close()
                            self.rwwD[rww_id]['rww'] = None

                        # rebuild RWW and put to resources
                        self._build_and_start_RWW(rww_id)
                        rww_ntasks[rww_id] = 0
                        resources.append(rww_id)

                        if self.restart_ex_tasks:
                            if self.log_exceptions:
                                self.logger.warning(f'>> putting task again to the tasks_que..')
                            tasks_que.appendleft((task_ix, task_try+1, task))

                        else:
                            if self.log_exceptions:
                                self.logger.warning(f'>> returning exception result..')

                            res_msg = QMessage(
                                type=   'exception_result',
                                data=   ResultOMPRException(exception_msg=msg.type, task=task))
                            if self.ordered_results: resultsD[task_ix] = res_msg
                            else: self.oque.put(res_msg)

                    # flush resultsD
                    while task_result_ix in resultsD:
                        self.oque.put(resultsD.pop(task_result_ix))
                        task_result_ix += 1

                if self.report_delay is not None and time.time()-iv_time > self.report_delay:
                    iv_speed = iv_n_tasks/((time.time()-iv_time)/60)
                    speed = speed_cache.upd(iv_speed)
                    if speed != 0:
                        if speed > 10:      speed_str = f'{int(speed)} tasks/min'
                        else:
                            if speed > 1:   speed_str = f'{speed:.1f} tasks/min'
                            else:           speed_str = f'{1 / speed:.1f} min/task'
                        n_tasks_left = len(tasks_que)
                        est = n_tasks_left / speed
                        n_total_tasks = n_tasks_processed + n_tasks_left
                        progress = n_tasks_processed / n_total_tasks
                        self.logger.info(f'> ({progress * 100:4.1f}% left:{n_tasks_left}/{n_total_tasks} {time.strftime("%H:%M:%S")}) speed: {speed_str}, EST:{est:.1f}min')
                    else: self.logger.info(f'> processing speed unknown yet..')
                    iv_time = time.time()
                    iv_n_tasks = 0
                    self.logger.debug(self._get_RWW_info())

        def after_exception_handle_run(self):
            self._kill_allRWW()
            self.logger.debug(f'> {self.ip_name} killed all RW after exception occurred')

        def get_num_RWW(self):
            return len(self.rwwD)

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
            devices: DevicesParam=          'all',
            name: str=                      'OMPRunner',
            ordered_results: bool=          True,   # returns results in the order of tasks
            task_timeout: Optional[float]=  None,   # (sec) timeout for task, if task processing time exceeds time, process will be killed and task not restarted, INFO: will then return result of type OMPRException
            restart_ex_tasks: bool=         True,   # restarts tasks that caused exception, INFO: for False may return result of type OMPRException
            log_exceptions: bool=           True,   # allows to log exceptions
            raise_RWW_exception: bool=      False,  # forces RWW to raise exceptions (.. all but KeyboardInterrupt)
            report_delay: Union[int,str]=   'auto', # num sec between speed_report, 'auto' uses loglevel, for 'none' there is no speed report
            logger=                         None,
            loglevel=                       20):

        self.omp_name = name

        if not logger:
            logger = get_pylogger(
                name=       self.omp_name,
                add_stamp=  False,
                folder=     None,
                level=      loglevel)
        self.logger = logger

        self.logger.info('*** OMPRunner *** inits..')
        self.logger.info(f'> name:     {self.omp_name}')
        self.logger.info(f'> pid:      {os.getpid()}')
        self.logger.info(f'> rw_class: {rw_class.__name__}')

        self._tasks_que = Que()             # que of tasks to be processed
        self._results_que = Que()           # que of ready results
        self._n_tasks_received: int = 0     # number of tasks received from user till now
        self._n_results_returned: int = 0   # number of results returned to user till now

        if report_delay == 'none': report_delay = None
        if report_delay == 'auto': report_delay = 30 if loglevel>10 else 10

        self._internal_processor = OMPRunner.InternalProcessor(
            ique=                   self._tasks_que,
            oque=                   self._results_que,
            rw_class=               rw_class,
            rw_init_kwargs=         rw_init_kwargs if rw_init_kwargs else {},
            rw_lifetime=            rw_lifetime,
            devices=                devices,
            ordered_results=        ordered_results,
            task_timeout=           task_timeout,
            restart_ex_tasks=       restart_ex_tasks,
            log_exceptions=         log_exceptions,
            raise_RWW_exception=    raise_RWW_exception,
            report_delay=           report_delay,
            logger=                 self.logger)
        self._internal_processor.start()

    # (not blocking) takes tasks for processing, starts processing, does not return anything
    def process(self, tasks: dict or List[dict]):
        if type(tasks) is dict: tasks = [tasks]
        self._tasks_que.put(QMessage(type='tasks', data=tasks))
        self._n_tasks_received += len(tasks)

    # returns single result, may block or not
    def get_result(self, block=True) -> Optional[Any]:
        if self._n_results_returned == self._n_tasks_received:
            self.logger.info(f'OMPRunner get_result() returns None since already returned all results (for all given tasks: n_results_returned == n_tasks_received)')
            return None
        else:
            if block:   msg = self._results_que.get()
            else:       msg = self._results_que.get_if()
            if msg:
                self._n_results_returned += 1
                return msg.data
            return None

    # returns results of all tasks put up to NOW
    def get_all_results(self, pop_ex_results=False) -> List[Any]:
        results = []
        n_results = self._n_tasks_received - self._n_results_returned
        while len(results) < n_results:
            results.append(self.get_result(block=True))
        if pop_ex_results:
            results = [r for r in results if type(r) is not ResultOMPRException]
        return results

    def get_tasks_stats(self) -> Dict[str,int]:
        return {
            'n_tasks_received':     self._n_tasks_received,
            'n_results_returned':   self._n_results_returned}

    def get_num_workers(self):
        return self._internal_processor.get_num_RWW()

    def exit(self):
        if self._n_results_returned != self._n_tasks_received:
            self.logger.warning(f'{self.omp_name} exits while not all results were returned to user!')
        self._internal_processor.exit()
        self.logger.info(f'> {self.omp_name}: internal processor stopped, {self.omp_name} exits.')
