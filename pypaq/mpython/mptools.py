from abc import ABC, abstractmethod
from multiprocessing import cpu_count, Process, Queue
from queue import Empty
import psutil
import time
from typing import Any, Optional

from pypaq.lipytools.pylogger import get_pylogger


# message sent between processes via Ques (my que)
class QMessage:
    def __init__(self, type:str, data:Any):
        self.type = type
        self.data = data

# my que
class Que:

    def __init__(self):
        self.q = Queue()

    def put(self, msg:QMessage, **kwargs):
        assert isinstance(msg, QMessage)
        self.q.put(msg, **kwargs)

    # does not raise Empty exception, but returns None in case no QMessage
    def get(self,
            block: bool=                True,
            timeout: Optional[float]=   None,
            ) -> Optional[QMessage]:
        try:
            msg = self.q.get(block=block, timeout=timeout)
            assert isinstance(msg, QMessage)
            return msg
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
            logger=                     None,
            loglevel=                   20):

        ABC.__init__(self)
        Process.__init__(self, target=self.__run)
        if id is None: id = self.name

        self.id = id

        if not logger:
            logger = get_pylogger(
                name=       self.id,
                add_stamp=  False,
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
            self.logger.info(f'> ExSubprocess ({self.id}, pid:{self.pid}) - starting subprocess_method()')
            self.subprocess_method()
            self.logger.info(f'> ExSubprocess ({self.id}, pid:{self.pid}) - finished subprocess_method()')
        except KeyboardInterrupt:
            self.__exception_handle('KeyboardInterrupt')
        except Exception as e:
            self.__exception_handle(f'other: {str(e)}')
            if self.raise_unk_exception: raise e

    # method run in subprocess, to be implemented
    @abstractmethod
    def subprocess_method(self): pass

    # when exception occurs, message with exception data is put on the output que
    def __exception_handle(self, name:str):
        msg = QMessage(
            type=   f'ex_{name}, ExSubprocess id: {self.id}, pid: {self.pid}',
            data=   self.id) # returns ID here to allow process identification
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