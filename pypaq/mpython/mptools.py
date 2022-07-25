from abc import ABC, abstractmethod
from multiprocessing import cpu_count, Process, Queue
from queue import Empty
import psutil
from typing import List, Any, Optional

"""
devices: DevicesParam - (parameter) manages GPUs, gives options for CPUs
    int                 - one (system) CUDA ID
    -1                  - last AVAILABLE CUDA
    None                - single CPU device
    str                 - device in TF format (e.g. '/device:GPU:0')
    [] (empty list)     - all AVAILABLE CUDA
    [int,-1,None,str]   - list of devices: ints (CUDA IDs), may contain None, possible repetitions
"""
DevicesParam: int or None or str or List[int or None or str] = -1

"""
multiproc: MultiprocParam - (parameter) manages CPUs only
    'auto'              - automatic
    'all'               - all CPU cores
    'off'               - single core == 1
    int (1-N)           - number of CPU cores
"""
MultiprocParam: str or int = 'auto'


# message sent between processes via Ques (my que)
class QMessage:
    def __init__(self, type:str, data:Any):
        self.type = type
        self.data = data

# my que
class Que:

    def __init__(self):
        self.q = Queue()

    def put(self, obj:QMessage, **kwargs):
        assert isinstance(obj, QMessage)
        self.q.put(obj, **kwargs)

    def get(self, **kwargs) -> QMessage:
        obj = self.q.get(**kwargs)
        assert isinstance(obj, QMessage)
        return obj

    def get_if(self) -> Optional[QMessage]:
        try:
            obj = self.q.get_nowait()
            assert isinstance(obj, QMessage)
            return obj
        except Empty:
            return None

    def empty(self): return self.q.empty()

    def qsize(self): return self.q.qsize()


# Exception Managed Subprocess - with basic exceptions management
class ExSubprocess(Process, ABC):

    def __init__(
            self,
            ique: Que,                          # input que
            oque: Que,                          # output que
            id: Optional[int or str]=   None,   # unique id to identify the subprocess, if not given takes from Process name
            raise_unk_exception=        True,   # raises exception other than KeyboardInterrupt
            verb: int=                  0):

        ABC.__init__(self)
        Process.__init__(self, target=self.__exm_run)
        if id is None: id = self.name

        self.id = id
        self.verb = verb
        self.raise_unk_exception = raise_unk_exception

        self.ique = ique
        self.oque = oque

        if self.verb>0: print(f'\n*** ExSubprocess ({self.id}) initialized')

    # core method run in subprocess
    @abstractmethod
    def subprocess_method(self): pass

    # when exception occurs, message with exception data is put on the output que
    def __exception_handle(self, name: str):
        msg = QMessage(
            type=   f'ex_{name}, ExSubprocess id: {self.id}, pid: {self.pid}',
            data=   self.id) # returns ID here to allow process identification
        self.oque.put(msg)
        if self.verb>0: print(f' > ExSubprocess ({self.id}) halted by exception: {name}')
        self.after_exception_handle_run()

    # this method may be implemented to run code by a self (Process) after exception occurred
    def after_exception_handle_run(self): pass

    # exception managed subprocess run (process target method), subprocess_method() will be executed till finish or exception
    def __exm_run(self):
        try:
            if self.verb>0: print(f' > ExSubprocess ({self.id}, pid:{self.pid}) - starting subprocess_method()')
            self.subprocess_method()
            if self.verb>0: print(f' > ExSubprocess ({self.id}, pid:{self.pid}) - finished subprocess_method()')
        except KeyboardInterrupt:
            self.__exception_handle('KeyboardInterrupt')
        except Exception as e:
            self.__exception_handle(f'other: {str(e)}')
            if self.raise_unk_exception: raise e

    def kill(self):
        if self.alive: super().kill()

    def terminate(self):
        if self.alive: super().terminate()

    @property
    def alive(self):
        if self.closed: return False
        return self.is_alive()

    @property
    def closed(self):
        return self._closed

    def get_info(self) -> str:
        pid = self.pid if not self.closed else '<pid:closed>'
        exitcode = self.exitcode if not self.closed else '<exitcode:closed>'
        nfo = f'{str(self)}, pid: {pid}, parent pid: {self._parent_pid}, alive: {self.alive}, exitcode: {exitcode}'
        return nfo


def sys_res_nfo():
    vm = psutil.virtual_memory()
    gb = 1024 ** 3
    return {
        'cpu_count':    cpu_count(),
        'mem_total_GB': vm.total / gb,
        'mem_used_%':   vm.percent}
