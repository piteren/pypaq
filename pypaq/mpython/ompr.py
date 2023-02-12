"""

    2021 (c) piteren

    OMPRunner - Object based Multi-Processing Runner

        Tasks may be given for OMPRunner with process() method - one dict after another or in packages (List[dict]).
        Results may be received with two get methods (single or all) and by default will be ordered with tasks order.
        Result returned by RunningWorker may be Any type.
        OMPRunner needs to be closed with exit().

        OMPRunner processes given tasks with InternalProcessor (IP) that guarantees non-blocking interface of OMPRunner.
        Tasks are processed by RunningWorker class objects that are managed by IP (and additionally wrapped by RWW).

        RunningWorker must be inherited and its process() implemented.
        process() takes task via **kwargs and returns task result.
        There are two main policies of RunningWorker lifecycle:
            1st - RunningWorker is closed after processing some task (1 typically but may be N)
            2nd - RunningWorker is closed only when crashes or with the OMP exit
            Each policy has job specific pros and cons. By default, second is activated with 'rw_lifetime=None'.
            'rw_lifetime=None' has some pros and cons:
                + all RunningWorkers are initialized once while OMP inits - it saves a time
                - memory kept by the RunningWorker may grow with the time (while processing many tasks)

"""

from abc import ABC, abstractmethod
from collections import deque
from inspect import getfullargspec
import os
import psutil
import signal
import time
from typing import Any, List, Dict, Optional, Union, Callable

from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.pms.base import get_params
from pypaq.mpython.devices import DevicesPypaq, get_devices
from pypaq.mpython.mptools import QMessage, Que, ExSubprocess


# Worker for tasks, to be implemented - processes task given with kwargs and returns result
class RunningWorker(ABC):

    @abstractmethod # processing method to be implemented
    def process(self, **kwargs) -> Any: pass

# OMPR Exception, also returned when task raises any exception while processed by RW
class OMPRException(Exception):

    def __init__(self, *args, task:Optional[Dict]=None):
        self.task = task
        Exception.__init__(self, *args)

# Object based Multi-Processing Runner
class OMPRunner:

        # Internal Processor of OMPRunner
    class InternalProcessor(ExSubprocess):

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

                # register an handler for the timeout
                def handler_timeout(signum, frame):
                    raise OMPRException('RW timeout')

                self.logger.info(f'> RWWrap ({self.id}) pid: {os.getpid()} inits RunningWorker')
                rwo = self.rw_class(**self.rw_init_kwargs)
                self.logger.info(f'> RWWrap ({self.id}) starts process loop..')

                signal.signal(signal.SIGALRM, handler_timeout) # register handler_timeout as a handler for the signal.SIGALRM function

                while True:
                    ompr_msg: QMessage = self.ique.get()
                    if ompr_msg.type == 'break':
                        break
                    if ompr_msg.type == 'hold_check':
                        self.oque.put(QMessage(type='hold_ready', data=None))
                    if ompr_msg.type == 'task':
                        task_ix = ompr_msg.data['task_ix']
                        timeout = ompr_msg.data['task_timeout']
                        task = ompr_msg.data['task']
                        result = None

                        # try block for timeout exception
                        try:
                            if timeout is not None: signal.alarm(timeout)

                            # try block for process exceptions
                            try:                   result = rwo.process(**task)
                            except Exception as e: result = OMPRException(f'exception while processing task {task_ix}: {e}', task=task)

                            if timeout is not None: signal.alarm(0)
                        except Exception as e:
                            result = OMPRException(f'exception while processing task {task_ix}: {e}', task=task)

                        finally:
                            self.oque.put(QMessage(
                                type=   'exception' if type(result) is OMPRException else 'result',
                                data=   {
                                    'rww_id':   self.id,
                                    'task_ix':  task_ix,
                                    'result':   result}))

                self.logger.info(f'> RWWrap (id: {self.id}) finished process loop')

        POISON_MSG = QMessage(type='poison', data=None)

        def __init__(
                self,
                ique: Que,                              # tasks and other messages
                oque: Que,                              # results and exception_results
                rw_class: type(RunningWorker),
                rw_init_kwargs: Optional[Dict],
                rw_lifetime: Optional[int],
                devices: DevicesPypaq,
                ordered_results: bool,
                task_timeout: Optional[int],
                log_RWW_exception: bool,
                raise_RWW_exception: bool,
                report_delay: Optional[int],
                logger):

            self.rw_class = rw_class
            self.ip_name = f'InternalProcessor_for_{self.rw_class.__name__}' # INFO: .name conflicts with Process.name

            self.logger = logger
            self.logger.info(f'*** {self.ip_name} *** inits..')

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

            devices = get_devices(devices=devices, torch_namespace=False)

            dev_param_name = None
            pms = getfullargspec(self.rw_class).args
            if 'devices' in pms: dev_param_name = 'devices'
            if 'device' in pms: dev_param_name = 'device'

            self.ordered_results = ordered_results
            self.task_timeout = task_timeout
            self.log_RWW_exception = log_RWW_exception
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
                logger=                 get_child(self.logger))
            self.rwwD[id]['rww'].start()
            self.logger.debug(f'> {self.ip_name} built and started RWWrap id: {id}..')


        def build_and_start_allRWW(self):
            self.logger.info(f'> {self.ip_name} is going to build and start {len(self.rwwD)} RunningWorkers..')
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
                ind = self.rwwD[id]['rww'].ique.get(block=False)
                if not ind: break
            self.rwwD[id]['rww'].join()
            self.rwwD[id]['rww'] = None
            self.logger.debug(f'> {self.ip_name} killed and joined RWWrap id: {id}..')


        def _kill_allRWW(self):
            self.logger.info(f'> {self.ip_name} is going to kill and join {len(self.rwwD)} RunningWorkers..')
            for id in self.rwwD:
                if self.rwwD[id]['rww'] is not None:
                    self._kill_RWW(id)
            self.logger.info(f'> {self.ip_name} killed and joined all RunningWorkers')

        # this method checks if all RunningWorkers are ready to process tasks
        def hold_till_allRWW_ready(self):
            self.logger.info(f'> hold: {self.ip_name} is checking RW readiness..')
            for id in self.rwwD:
                if self.rwwD[id]['rww'] is None:
                    self.logger.warning('some RWW are not started, cannot hold!!!')
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
            s = f'# {self.ip_name} mem: {ip_mem}MB, omp+sp/used: {tot_mem/1024:.1f}/{used:.1f}GB ({int(vm.percent)}%VM) '
            if len(rww_mem) > 6: s += f'subproc: {rww_mem[:3]}-{int(sum(rww_mem)/len(rww_mem))}-{rww_mem[-3:]} ({alive_info})'
            else:                s += f'subproc: {rww_mem} ({alive_info})'
            return s

        # main loop of InternalProcessor (run by ExSubprocess)
        def subprocess_method(self):

            self.logger.info(f'> {self.ip_name} (pid: {os.getpid()}) starts loop with {len(self.rwwD)} RWW')
            self.build_and_start_allRWW()

            next_task_ix = 0                                # next task index (index of task that will be processed next)
            task_result_ix = 0                              # index of task result that should be put to self.results_que now
            rww_ntasks = {k: 0 for k in self.rwwD.keys()}   # number of tasks processed by each RWW since restart

            iv_time = time.time()                           # interval report time
            iv_n_tasks = 0                                  # number of tasks finished since last interval
            n_tasks_processed = 0                           # number of tasks processed (total)
            speed_mavg = MovAvg(factor=0.2)                 # speed (tasks/min) moving average

            tasks_que = deque()                             # que of (task_ix, task) to be processed (received from the self.ique)
            resources = list(self.rwwD.keys())              # list [rww_id] of all available (not busy) resources

            resultsD: Dict[int, Any] = {}                   # results dict {task_ix: result(data)} for ordered tasks
            while True:

                msg_ique = self.ique.get(block=False) # get message from the ique

                # block when no message, no tasks present and all RWW waiting
                if not msg_ique and not tasks_que and len(resources) == len(self.rwwD):
                    msg_ique = self.ique.get()

                # process ique message
                if msg_ique:

                    if msg_ique.type not in ['tasks','poison']:
                        nfo = f'{self.ip_name} received unknown message type: \'{msg_ique.type}\''
                        self.logger.error(nfo)
                        raise OMPRException(nfo)

                    self.logger.debug(f'> {self.ip_name} got \'{msg_ique.type}\' message from ique')
                    if msg_ique.type == 'tasks':
                        # unpack tasks
                        for task in msg_ique.data:
                            tasks_que.append((next_task_ix, task))
                            next_task_ix += 1
                    if msg_ique.type == 'poison':
                        # all RWW have to be killed here, from the loop
                        # we want to kill them because it is quicker than waiting for them till finish tasks
                        # - we do not need their results anymore
                        self._kill_allRWW()
                        break # break the loop

                # put resources into work
                while resources and tasks_que:
                    self.logger.debug(f'> free resources: {len(resources)}, tasks_que len: {len(tasks_que)}, ique.qsize: {self.ique.qsize()}')

                    rww_id = resources.pop(0) # take first free resource

                    # eventually restart RWW (lifetime limit)
                    if self.rw_lifetime and rww_ntasks[rww_id] >= self.rw_lifetime:
                        self.logger.debug(f'> restarting RWWrap id: {rww_id} because of lifetime condition..')
                        self._kill_RWW(rww_id)
                        self._build_and_start_RWW(rww_id)
                        rww_ntasks[rww_id] = 0

                    # get first task, prepare and put message for RWWrap
                    task_ix, task = tasks_que.popleft()
                    msg = QMessage(
                        type=   'task',
                        data=   {
                            'task_ix':      task_ix,
                            'task_timeout': self.task_timeout,
                            'task':         task})
                    self.rwwD[rww_id]['rww'].ique.put(msg)

                    self.logger.debug(f'> put task {task_ix} for RWWrap({rww_id})')

                # manage RWW results
                msg_rww = self.que_RW.get() # wait for one
                while True:

                    if msg_rww.type not in ['result','exception']:
                        self.logger.warning(f'> {self.ip_name} got from RWW unknown message type: \'{msg_rww.type}\'')
                        break

                    rww_id =    msg_rww.data['rww_id']
                    task_ix =   msg_rww.data['task_ix']
                    result =    msg_rww.data['result']
                    self.logger.debug(f'> {self.ip_name} got message from RWW {rww_id} for task {task_ix}')

                    if type(result) is OMPRException and self.log_RWW_exception:
                        self.logger.warning(f'> {self.ip_name} got exception message from RWW {rww_id} for task {task_ix}: {result}')

                    res_msg = QMessage(type='result', data=result)
                    if self.ordered_results: resultsD[task_ix] = res_msg
                    else: self.oque.put(res_msg)

                    rww_ntasks[rww_id] += 1
                    resources.append(rww_id)

                    n_tasks_processed += 1
                    iv_n_tasks += 1

                    # eventually get next
                    msg_rww = self.que_RW.get(block=False)
                    if not msg_rww: break

                # flush resultsD
                while task_result_ix in resultsD:
                    self.oque.put(resultsD.pop(task_result_ix))
                    task_result_ix += 1

                if self.report_delay is not None and time.time()-iv_time > self.report_delay:

                    iv_speed = iv_n_tasks/((time.time()-iv_time)/60)
                    speed = speed_mavg.upd(iv_speed)

                    if speed != 0:
                        if speed > 10:      speed_str = f'{int(speed)} tasks/min'
                        else:
                            if speed > 1:   speed_str = f'{speed:.1f} tasks/min'
                            else:           speed_str = f'{1 / speed:.1f} min/task'
                        n_tasks_left = len(tasks_que)
                        est = n_tasks_left / speed
                        progress = n_tasks_processed / next_task_ix
                        self.logger.info(f'> progress: {progress * 100:4.1f}% speed: {speed_str}, left:{n_tasks_left}/{next_task_ix}, EST:{est:.1f}min')
                    else: self.logger.info(f'> processing speed unknown yet..')

                    iv_time = time.time()
                    iv_n_tasks = 0

                    self.logger.debug(self._get_RWW_info())
                    self.logger.debug(f'rww_ntasks:')
                    for rk in sorted(rww_ntasks.keys()):
                        self.logger.debug(f'{rk:2}: {rww_ntasks[rk]}')


        def after_exception_handle_run(self):
            self._kill_allRWW()
            self.logger.debug(f'> {self.ip_name} killed all RWW after exception occurred')


        def get_num_RWW(self):
            return len(self.rwwD)

        # method to call out of the process (to exit it)
        def exit(self) -> None:

            if self.alive:
                self.ique.put(self.POISON_MSG)

                while self.alive:
                    # flush the oque
                    while True:
                        res = self.oque.get(block=False)
                        if res is None: break
                    self.join(timeout=0.0001)


    def __init__(
            self,
            rw_class: type(RunningWorker),              # RunningWorker class that will run() given tasks
            rw_init_kwargs: Optional[Dict]= None,       # RunningWorker __init__ kwargs, logger is managed by OMPRunner
            rw_lifetime: Optional[int]=     None,       # RunningWorker lifetime, for None or 0 is unlimited, for N <1,n> each RW will be restarted after processing N tasks
            devices: DevicesPypaq=          'all',
            name: str=                      'OMPRunner',
            ordered_results: bool=          True,       # returns results in the order of tasks
            task_timeout: Optional[int]=    None,       # (sec)  RW process will be killed after that time of processing, OMPRException will be returned as a task result
            log_RWW_exception: bool=        True,       # logs RWW exceptions
            raise_RWW_exception: bool=      False,      # forces RWW to raise exceptions (.. all but KeyboardInterrupt)
            report_delay: Union[int,str]=   'auto',     # num sec between speed_report, 'auto' uses loglevel, for 'none' there is no speed report
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

        if self.logger.level < 20:
            log_RWW_exception = True

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

        if not rw_init_kwargs:
            rw_init_kwargs = {}

        # eventually add self.logger to rw_init_kwargs
        rw_class_params = get_params(rw_class.__init__)
        if 'logger' in rw_class_params['with_defaults'] or 'logger' in rw_class_params['without_defaults']:
            rw_init_kwargs['logger'] = self.logger

        self._internal_processor = OMPRunner.InternalProcessor(
            ique=                   self._tasks_que,
            oque=                   self._results_que,
            rw_class=               rw_class,
            rw_init_kwargs=         rw_init_kwargs if rw_init_kwargs else {},
            rw_lifetime=            rw_lifetime,
            devices=                devices,
            ordered_results=        ordered_results,
            task_timeout=           task_timeout,
            log_RWW_exception=      log_RWW_exception,
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
            else:       msg = self._results_que.get(block=False)
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
            results = [r for r in results if type(r) is not OMPRException]
        return results

    def get_tasks_stats(self) -> Dict[str,int]:
        return {
            'n_tasks_received':     self._n_tasks_received,
            'n_results_returned':   self._n_results_returned}

    def get_num_workers(self) -> int:
        return self._internal_processor.get_num_RWW()

    def exit(self):
        if self._n_results_returned != self._n_tasks_received:
            self.logger.warning(f'{self.omp_name} exits while not all results were returned to user!')
        self._internal_processor.exit()
        self.logger.info(f'> {self.omp_name}: internal processor stopped, {self.omp_name} exits.')

# function to simple process tasks with function using OMPR
def simple_process(
        tasks: List[Dict],      # tasks to process
        function: Callable,     # processing function
        num_workers: int=   4,
        **kwargs,
) -> List[Any]:

    class SimpleRW(RunningWorker):
        def process(self, **kwargs) -> Any:
            return function(**kwargs)

    ompr = OMPRunner(
        rw_class=   SimpleRW,
        devices=    [None]*num_workers,
        **kwargs)

    ompr.process(tasks)
    results = ompr.get_all_results()
    ompr.exit()
    return results