import inspect
from typing import Any, Dict, List, Tuple, Callable, Optional

from pypaq.lipytools.printout import float_to_str

AXIS =  str                                     # axis type (parameter name)
P_VAL = float or int or Any                     # point value (parameter value)
POINT = Dict[AXIS, P_VAL]                       # POINT ia a dict {parameter: value}


"""
    PSDD - Parameters Space Definition Dict
        dictionary that defines space of POINTS {axis(parameter name): list or tuple or value}
            - list of ints or floats defines continuous range
            - tuple may contain elements of any type (even non-numeric), may have one
            - any other type is considered to be a constant (single value)

            example:
            {   'a':    [0.0, 1],               # range of floats
                'b':    (-1,-7,10,15.5,90,30),  # set of num(float) values, num will be sorted
                'c':    ('tat','mam','kot'),    # set of diff values
                'd':    [0,10],                 # range of ints
                'e':    (-2.0,2,None)}          # set of diff values
                'f':    (16.2,)}                # single value

"""

RANGE = List[P_VAL] or Tuple[P_VAL] or P_VAL    # axis range type (range of parameter)
PSDD  = Dict[AXIS, RANGE]                       # Parameters Space Definition Dict {parameter: range}}


# prepares nice string of POINT
def point_str(p: POINT) -> str:
    s = '{'
    for axis in sorted(list(p.keys())):
        val = p[axis]
        vs = float_to_str(val) if type(val) is float else str(val)
        s += f'{axis}:{vs} '
    s = s[:-1] + '}'
    return s

# prepares function parameters dictionary
def get_params(function: Callable) -> Dict:

    params_dict = {
        'without_defaults': [],
        'with_defaults':    {}}

    if function:

        specs = inspect.getfullargspec(function)
        #print(specs)

        params = specs.args + specs.kwonlyargs

        vals = []
        if specs.defaults:
            vals += list(specs.defaults)
        if specs.kwonlydefaults:
            vals += list(specs.kwonlydefaults.values())

        while len(params) > len(vals):
            params_dict['without_defaults'].append(params.pop(0))

        params_dict['with_defaults'] = {k: v for k,v in zip(params,vals)}

    return params_dict

# prepares sub-POINT trimmed to function params (given wider POINT)
def point_trim(
        func: Optional[Callable],
        point: POINT,
        remove_self= True # removes self in case of methods (class)
) -> POINT:
    if func is None: return {}
    pms = get_params(func)
    valid_keys = pms['without_defaults'] + list(pms['with_defaults'].keys())
    if remove_self and 'self' in valid_keys: valid_keys.remove('self')
    func_dna = {k: point[k] for k in point if k in valid_keys} # filter to get only params accepted by func
    return func_dna
