from functools import wraps
from inspect import getfullargspec
import time

from pypaq.lipytools.printout import stamp
from pypaq.pms.base import get_params


# decorator printing execution time report
def timing(f):
    def new_f(*args, **kwargs):
        stime = time.time()
        ret = f(*args, **kwargs)
        taken_sec = time.time() - stime
        taken = f'{taken_sec/60:.1f}min' if taken_sec > 100 else f'{taken_sec:.1f}sec'
        print(f'(@timing:{stamp(letters=None)}, taken {taken}) {f.__name__} finished')
        return ret
    new_f.__name__ = f'{f.__name__}:@timing'
    return new_f

# prints debug info about parameters and given args/kwargs of function/method
def args(f):
    def new_f(*args, **kwargs):
        ins = getfullargspec(f)
        no_val = '--'

        arL = ins.args
        if arL[0] == 'self': arL = arL[1:]                  # simple, BUT not 100% accurate, ..but who names param 'self'?

        defL = list(ins.defaults) if ins.defaults else []   # list of default values
        defL = [no_val]*(len(arL)-len(defL)) + defL         # pad them with no_val
        arL = [list(e) for e in zip(arL,defL)]              # zip together

        # append args from the beginning
        if args:
            arvL = args[1:]
            for ix in range(len(arvL)):
                v = arvL[ix]
                arL[ix].append(v)

        kwL = [[k,kwargs[k]] for k in kwargs]               # kwargs in list

        # get from kwargs params of f, append their values and remove from kwargs
        kremIXL = []
        for ix in range(len(kwL)):
            for e in arL:
                if kwL[ix][0]==e[0]:
                    kremIXL.append(ix)
                    e.append(kwL[ix][1])
        for ix in reversed(kremIXL): kwL.pop(ix)

        # add no value to not overridden params
        for e in arL:
            if len(e)<3: e.append(no_val)

        # calc columns widths, trim if
        kw = [10,10,10]
        for e in arL:
            for i in range(3):
                lse = len(str(e[i]))
                if lse>kw[i]: kw[i] = lse
        for e in kwL:
            for i in range(2):
                lse = len(str(e[i]))
                ki = i if not i else 2
                if lse > kw[ki]: kw[ki] = lse
        while sum(kw) > 120:
            mx = max(kw)
            for ix in range(3):
                if kw[ix] == mx: kw[ix] -= 1
        mx = max(kw)

        def s(p):
            r = str(p)
            if len(r) > mx: r = r[:mx-2] + '..'
            return r

        print(f'\n@args report: *****************************************************************************************')
        print(f'function __name__: {f.__name__}')
        print(f' > {"param":{str(kw[0])}s}  {"default":{str(kw[1])}s}  {"given":{str(kw[2])}s}')
        for e in arL:
            print(f'   {s(e[0]):{str(kw[0])}s}  {s(e[1]):{str(kw[1])}s}  {s(e[2]):{str(kw[2])}s}')
        if kwL: print(f' > **kwargs (not used by {f.__name__}):')
        for e in kwL:
            print(f'   {s(e[0]):{str(kw[0])}s}  {"":{str(kw[1])}s}  {s(e[1]):{str(kw[2])}s}')
        print('@args report finished *********************************************************************************\n')

        return f(*args, **kwargs)
    new_f.__name__ = f'{f.__name__}:@args'
    return new_f

# decorator for __init__(), automatically assigns __init__ args and kwargs to the object (self)
def autoinit(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):

        params = get_params(func)
        names = params['without_defaults'][1:]  # ..trim self
        defaults = params['with_defaults']

        for name, arg in list(zip(names, args)) + list(kwargs.items()):
            setattr(self, name, arg)

        # case, when default is given with args
        left_args = args[len(names):]
        for la,key in zip(left_args,defaults.keys()):
            defaults[key] = la

        for name in defaults:
            val = defaults[name]
            if not hasattr(self, name):
                setattr(self, name, val)

        func(self, *args, **kwargs)

    return wrapper