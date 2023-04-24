from multiprocessing import Process, Queue
from functools import partial


# non-blocking process, does not return anything
def proc(f):

    def new_f(*args, **kwargs):
        Process(target=partial(f, *args, **kwargs)).start()

    new_f.__name__ = f'@proc:{f.__name__}'
    return new_f

# blocking process, does not return anything
def proc_wait(f):

    def new_f(*args, **kwargs):
        p = Process(target=partial(f, *args, **kwargs))
        p.start()
        p.join()

    new_f.__name__ = f'@proc_wait:{f.__name__}'
    return new_f

# helper class (process with que - puts target result on que)
class MProc(Process):

    def __init__(
            self,
            que :Queue,
            f,
            args,
            kwargs):

        super().__init__(target=self.proc_m)
        self.que = que
        self.f = f
        self.ag = args
        self.kw = kwargs

    def proc_m(self):
        res = self.f(*self.ag, **self.kw)
        self.que.put(res)

# blocking process with return
def proc_return(f):
    def new_f(*args, **kwargs):
        que = Queue()
        p = MProc(que=que, f=f, args=args, kwargs=kwargs)
        p.start()
        ret = que.get()
        return ret
    new_f.__name__ = f'@proc_return:{f.__name__}'
    return new_f

# non-blocking process with return via que
def proc_que(que):
    def wrap(f):

        def new_f(*args, **kwargs):
            p = MProc(que=que, f=f, args=args, kwargs=kwargs)
            p.start()

        new_f.__name__ = f'@qproc:{f.__name__}'
        return new_f

    return wrap