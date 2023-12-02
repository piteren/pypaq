from multiprocessing import Process, Queue
from functools import partial


def proc(f):
    """ non-blocking process
    does not return anything """

    def new_f(*args, **kwargs):
        p = Process(target=partial(f, *args, **kwargs))
        p.start()

    new_f.__name__ = f'@proc:{f.__name__}'
    return new_f

def proc_wait(f):
    """ blocking process
    does not return anything """

    def new_f(*args, **kwargs):
        p = Process(target=partial(f, *args, **kwargs))
        p.start()
        p.join()

    new_f.__name__ = f'@proc_wait:{f.__name__}'
    return new_f


class MProc(Process):
    """ helper class
    process with que: puts target result on que """

    def __init__(self, que:Queue, f, args, kwargs):
        super().__init__(target=self.proc_m)
        self.que = que
        self.f = f
        self.ag = args
        self.kw = kwargs

    def proc_m(self):
        res = self.f(*self.ag, **self.kw)
        self.que.put(res)

def proc_return(f):
    """ blocking, returning process """

    def new_f(*args, **kwargs):
        que = Queue()
        p = MProc(que=que, f=f, args=args, kwargs=kwargs)
        p.start()
        ret = que.get()
        return ret
    new_f.__name__ = f'@proc_return:{f.__name__}'

    return new_f

def proc_que(que):
    """ non-blocking process
    returns (via given que) """

    def wrap(f):

        def new_f(*args, **kwargs):
            p = MProc(que=que, f=f, args=args, kwargs=kwargs)
            p.start()

        new_f.__name__ = f'@qproc:{f.__name__}'
        return new_f

    return wrap