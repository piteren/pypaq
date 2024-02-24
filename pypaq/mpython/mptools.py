from abc import ABC, abstractmethod
from multiprocessing import cpu_count, Process, Queue, Value
from queue import Empty
import psutil
import time
from typing import Any, Optional

from pypaq.exception import PyPaqException
from pypaq.lipytools.pylogger import get_pylogger


class MPythonException(PyPaqException):
    pass


# message sent between processes via Ques (my que)
class QMessage:
    def __init__(self, type:str, data:Optional[Any]=None):
        self.type = type
        self.data = data

    def __str__(self):
        return f'{self.__class__.__name__}, type:{self.type}, data:{self.data}'


# https://github.com/vterron/lemon/commit/9ca6b4b1212228dbd4f69b88aaf88b12952d7d6f
class SharedCounter:

    def __init__(self, n:int=0):
        self.count = Value('i', n)

    def increment(self, n:int=1):
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self) -> int:
        return self.count.value


# my que
class Que:

    def __init__(self):
        self.q = Queue()
        self.size = SharedCounter(0)

    def put(self, msg:QMessage, **kwargs):
        if not isinstance(msg, QMessage):
            raise MPythonException(f'\'msg\' should be type of QMessage, but is {type(msg)}')
        self.size.increment(1)
        self.q.put(msg, **kwargs)

    # does not raise Empty exception, but returns None in case no QMessage
    def get(self,
            block: bool=                True,
            timeout: Optional[float]=   None,
            ) -> Optional[QMessage]:
        try:
            msg = self.q.get(block=block, timeout=timeout)
            self.size.increment(-1)
            if not isinstance(msg, QMessage):
                raise MPythonException(f'\'msg\' should be type of QMessage, but is {type(msg)}')
            return msg
        except Empty:
            return None

    def empty(self) -> bool:
        return not self.qsize()

    def qsize(self) -> int:
        return self.size.value


# Exception Managed Subprocess - with basic exceptions management
class ExSubprocess(Process, ABC):

    def __init__(
            self,
            ique: Optional[Que]=        None,   # input que
            oque: Optional[Que]=        None,   # output que
            id: Optional[int or str]=   None,   # unique id to identify the subprocess, if not given takes from Process name
            raise_unk_exception=        True,   # raises exception other than KeyboardInterrupt
            logger=                     None,
            loglevel=                   30):

        super().__init__(target=self.__run)

        if id is None:
            id = self.name
        self.id = id

        if not logger:
            logger = get_pylogger(
                name=       self.id,
                folder=     None,
                level=      loglevel)
        self.logger = logger

        self.raise_unk_exception = raise_unk_exception

        self.ique = ique
        self.oque = oque

        self.logger.info(f'*** ExSubprocess *** id: {self.id} initialized')

    # process target method, wraps subprocess_method() with try / except
    def __run(self):
        try:
            self.logger.debug(f'> ExSubprocess ({self.id}, pid:{self.pid}) - started subprocess_method()')
            self.subprocess_method()
            self.logger.debug(f'> ExSubprocess ({self.id}, pid:{self.pid}) - finished subprocess_method()')
        except KeyboardInterrupt:
            self.__exception_handle('KeyboardInterrupt')
        except Exception as e:
            self.__exception_handle(f'other: {e}')
            if self.raise_unk_exception: raise e

    # method run in a subprocess, to be implemented
    @abstractmethod
    def subprocess_method(self): pass

    # when exception occurs, message with exception data is put on the output que
    def __exception_handle(self, name:str):
        msg = QMessage(
            type=   f'ex_{name}, ExSubprocess id: {self.id}, pid: {self.pid}',
            data=   self.id) # returns ID here to allow process identification
        if self.oque is not None:
            self.oque.put(msg)
        self.logger.warning(f'> ExSubprocess ({self.id}) halted by exception: {name}')
        self.after_exception_handle_run()

    # this method may be implemented and will be run after exception occurred
    def after_exception_handle_run(self): pass

    def kill(self):
        if self.alive: super().kill()
        while self.alive: time.sleep(0.01)

    def terminate(self):
        if self.alive: super().terminate()
        while self.alive: time.sleep(0.01)

    @property
    def alive(self):
        if self.closed: return False
        return self.is_alive()

    @property
    def closed(self):
        return self._closed

    # some process info
    def get_info(self) -> str:
        pid = self.pid if not self.closed else None
        pid_nfo = pid if pid is not None else '<pid:closed>'
        exitcode = self.exitcode if not self.closed else '<exitcode:closed>'
        mem_nfo = int(psutil.Process(pid).memory_info().rss / 1024 ** 2) if not exitcode else '-'
        nfo = f'{str(self)}, pid: {pid_nfo}, mem:{mem_nfo}MB, parent pid: {self._parent_pid}, alive: {self.alive}, exitcode: {exitcode}'
        return nfo


def sys_res_nfo():
    vm = psutil.virtual_memory()
    gb = 1024 ** 3
    return {
        'cpu_count':    cpu_count(),
        'cpu_used_%':   psutil.cpu_percent(interval=5), # over last 5 sec
        'mem_total_GB': vm.total / gb,
        'mem_used_%':   vm.percent}