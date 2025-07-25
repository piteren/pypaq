from abc import ABC, abstractmethod
from multiprocessing import cpu_count, Process, Queue, Value
import psutil
from queue import Empty
import time
from typing import Any, Optional, Union

from pypaq.exception import PyPaqException
from pypaq.lipytools.pylogger import get_pylogger


class MPythonException(PyPaqException):
    pass


class QMessage:
    """ QMessage is a type of object (message) sent between processes with Ques """
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


class Que:
    """ MultiProcessing Queue that:
    - manages its size
    - accepts messages of QMessage type """

    def __init__(self):
        self.q = Queue()
        self.size = SharedCounter(0)

    def put(self, msg:QMessage, **kwargs):
        if not isinstance(msg, QMessage):
            raise MPythonException(f'\'msg\' should be type of QMessage, but is {type(msg)}')
        self.size.increment(1)
        self.q.put(msg, **kwargs)

    def get(self, block:bool=True, timeout:Optional[float]=None) -> Optional[QMessage]:
        try:
            msg = self.q.get(block=block, timeout=timeout)
            self.size.increment(-1)
        except Empty:
            return None
        if not isinstance(msg, QMessage):
            raise MPythonException(f'\'msg\' should be type of QMessage, but is {type(msg)}')
        return msg

    @property
    def empty(self) -> bool:
        return not self.qsize

    @property
    def qsize(self) -> int:
        return self.size.value


class ExProcess(Process, ABC):
    """ Exception Managed Process
    implements basic exceptions management """

    def __init__(
            self,
            ique: Optional[Que]=            None,
            oque: Optional[Que]=            None,
            name: Optional[Union[str,int]]= None,
            raise_KeyboardInterrupt=        False,
            raise_Exception=                False,
            logger=                         None,
            loglevel=                       30,
    ):
        """
        :param ique:
            input Que, not used by this class, but may be used by child Classes
        :param oque:
            output Que, sends messages from the ExProcess,
            when Exception or KeyboardInterrupt occurs message is sent here
        :param name:
            identifies ExProcess, for None is taken from Process.name
        :param raise_KeyboardInterrupt:
            results in immediate exit after KeyboardInterrupt,
            without running post exception handler
        :param raise_Exception:
            raises Exception, for debug """

        super().__init__(target=self.__run)

        if name is not None:
            self.name = str(name)

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                folder=     None,
                level=      loglevel)
        self.logger = logger

        self.ique = ique
        self.oque = oque
        self.raise_KeyboardInterrupt = raise_KeyboardInterrupt
        self.raise_Exception = raise_Exception

        self.logger.info(f'*** {self.name} (ExProcess) *** initialized')

    def __run(self):
        """ process target method,
        wraps exprocess_method() with try / except = exception handling """
        try:
            self.logger.debug(f'> ExProcess ({self.name}, pid:{self.pid}) - started exprocess_method()')
            self.exprocess_method()
            self.logger.debug(f'> ExProcess ({self.name}, pid:{self.pid}) - finished exprocess_method()')

        except KeyboardInterrupt:

            if self.raise_KeyboardInterrupt:
                raise KeyboardInterrupt

            self.__exception_handle('KeyboardInterrupt')

            if self.raise_Exception:
                raise KeyboardInterrupt

        except Exception as e:

            self.__exception_handle(str(e))

            if self.raise_Exception:
                raise e

    @abstractmethod
    def exprocess_method(self):
        """ method run in a process, to be implemented """
        pass

    def __exception_handle(self, e_name:str):
        """ when exception occurs, message with exception data is put on the output que """
        if self.oque is not None:
            self.oque.put(QMessage(
                type=   f'Exception:{e_name}, ExProcess {self.name} (pid:{self.pid})',
                data=   self.name)) # returns name here to allow process identification
        self.logger.warning(f'> ExProcess ({self.name}) halted by Exception:{e_name}')
        self.after_exception_handle_run()

    def after_exception_handle_run(self):
        """ this method may be implemented and will be run after inside exception handler """
        pass

    @property
    def alive(self):
        """ not spawned process is not alive and cannot be joined,
        when proces alive: True->False it may be join()-ed """
        if self.closed:
            return False
        return self.is_alive()

    def terminate(self):
        if self.alive:
            super().terminate()
        while self.alive:
            time.sleep(0.01)

    def kill(self):
        if self.alive:
            super().kill()
        while self.alive:
            time.sleep(0.01)

    @property
    def closed(self):
        return self._closed

    def __str__(self):
        pid = self.pid if not self.closed else '<-closed->'
        exitcode = self.exitcode if not self.closed else '<-closed->'
        mem_nfo = '---'
        if self.alive:
            mem_mb = int(psutil.Process(self.pid).memory_info().rss / 1024 ** 2)
            mem_nfo = f'{mem_mb}MB'
        return (f'{super().__str__()}\n'
                f'> pid:            {pid}\n'
                f'> parent(pid):    {self._parent_pid}\n'
                f'> alive:          {self.alive}\n'
                f'> closed:         {self.closed}\n'
                f'> exitcode:       {exitcode}\n'
                f'> mem:            {mem_nfo}')


def sys_res_nfo():
    vm = psutil.virtual_memory()
    gb = 1024 ** 3
    return {
        'cpu_count':    cpu_count(),
        'cpu_used_%':   psutil.cpu_percent(interval=5), # over last 5 sec
        'mem_total_GB': vm.total / gb,
        'mem_used_%':   vm.percent}