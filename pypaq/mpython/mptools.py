from multiprocessing import cpu_count, Process, Queue, Value
import psutil
from queue import Empty
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


class SharedCounter:
    """ Process-safe integer counter backed by multiprocessing.Value (shared memory).
    Uses lock for atomic updates, so all processes see consistent value.
    Used by Que to track queue size reliably across processes and platforms """

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
    - has size
    - works with QMessage type messages """

    def __init__(self):
        self._q = Queue()
        self._size = SharedCounter(0)

    def put(self, msg:QMessage, **kwargs):
        if not isinstance(msg, QMessage):
            raise MPythonException(f'\'msg\' should be type of QMessage, but is {type(msg)}')
        self._size.increment(1)
        self._q.put(msg, **kwargs)

    def get(self, block:bool=True, timeout:Optional[float]=None) -> Optional[QMessage]:
        try:
            msg = self._q.get(block=block, timeout=timeout)
            self._size.increment(-1)
        except Empty:
            return None
        return msg

    @property
    def empty(self) -> bool:
        return not self.size

    @property
    def size(self) -> int:
        return self._size.value


class ExProcess(Process):
    """ Exception Managed Process
    implements basic exceptions management.
    Similar to Process, it should be started with start() """

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
        ique:
            input Que, not used by this class, but may be used by child Classes
        oque:
            output Que, sends messages from the ExProcess,
            when Exception or KeyboardInterrupt occurs message is sent here
        name:
            identifies ExProcess, for None is taken from Process.name
        raise_KeyboardInterrupt:
            results in immediate exit after KeyboardInterrupt,
            without running post exception handler
        raise_Exception:
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
        """ process target method """
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

    def exprocess_method(self):
        """ method run in a process, to be implemented,
        this method is run inside try / except statement to hande ExProcess exceptions """
        raise NotImplementedError

    def __exception_handle(self, e_name:str):
        """ when exception occurs, message with exception data is put on the output que """
        if self.oque is not None:
            # returns self.name as data to allow process identification
            self.oque.put(QMessage(
                type=   f'Exception:{e_name}, ExProcess {self.name} (pid:{self.pid})',
                data=   self.name))
        self.logger.warning(f'> ExProcess ({self.name}) halted by Exception:{e_name}')
        self.after_exception_handle_run()

    def after_exception_handle_run(self):
        """ this method may be implemented and will be run
        after exception occurred inside exception handler """
        pass

    def kill_and_close(self):
        self.kill()
        self.join()
        self.close()

    @property
    def alive(self) -> bool:
        return not getattr(self, '_closed', False) and self.is_alive()

    @property
    def mem_usage(self) -> int:
        mem = 0
        if self.alive:
            try:
                mem = int(psutil.Process(self.pid).memory_info().rss / 1024 ** 2)
            except (psutil.NoSuchProcess, ProcessLookupError):
                pass
        return mem


def sys_res_nfo():
    vm = psutil.virtual_memory()
    gb = 1024 ** 3
    return {
        'cpu_count':    cpu_count(),
        'cpu_used_%':   psutil.cpu_percent(interval=5), # over last 5 sec
        'mem_total_GB': vm.total / gb,
        'mem_used_%':   vm.percent}