import inspect
from typing import Any, Dict, List, Tuple, Callable, Optional, Union, Type

from pypaq.exception import PyPaqException
from pypaq.lipytools.printout import float_to_str

AXIS =  str                     # axis type (parameter name)
P_VAL = Union[float, int, Any]  # point value (parameter value)
POINT = Dict[AXIS, P_VAL]       # POINT ia a dict {parameter: value}


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


class PMSException(PyPaqException):
    pass


def point_str(p:POINT) -> str:
    """ prepares nice string of POINT """

    avL = []
    for axis in sorted(list(p.keys())):

        val = p[axis]

        val_str = ''
        if type(val) is float:
            val_str = float_to_str(val)
        if type(val) is bool:
            val_str = f'{str(val):5}'
        if not val_str:
            val_str = str(val)

        avL.append(f'{axis}:{val_str}')

    return '{' + ' '.join(avL) + '}'


def get_params(function:Callable) -> Dict:
    """ prepares function parameters dictionary """

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


def get_class_init_params(cl:Type) -> Dict:
    """ prepares class.__init__ parameters dictionary
    recurrently goes down to base classes """

    def _update_params_dict(
            without_defaults: List,
            with_defaults: Dict):
        """ in-place updates current params_dict """

        # cross-remove
        for k in without_defaults:
            if k in params_dict['with_defaults']:
                params_dict['with_defaults'].pop(k)
        for k in with_defaults:
            if k in params_dict['without_defaults']:
                params_dict['without_defaults'].remove(k)

        for p in without_defaults:
            if p not in params_dict['without_defaults']:
                params_dict['without_defaults'].append(p)
        params_dict['with_defaults'].update(with_defaults)

    params_dict = {
        'without_defaults': [],
        'with_defaults':    {}}

    bases = list(cl.__bases__)
    if object in bases:
        bases.remove(object)

    for bcl in bases:
        bcl_pd = get_class_init_params(bcl)
        _update_params_dict(
            without_defaults=   bcl_pd['without_defaults'],
            with_defaults=      bcl_pd['with_defaults'])

    cl_pd = get_params(cl.__init__)
    _update_params_dict(
        without_defaults=   cl_pd['without_defaults'],
        with_defaults=      cl_pd['with_defaults'])

    return params_dict


def point_trim(
        fc: Optional[Union[Callable,Type]],
        point: POINT,
        remove_self= True,  # removes self in case of class methods
) -> POINT:
    """ prepares sub-POINT trimmed to function params (given wider POINT) """

    if fc is None:
        return {}

    pms = get_params(fc) if inspect.isfunction(fc) else get_class_init_params(fc)

    valid_keys = pms['without_defaults'] + list(pms['with_defaults'].keys())

    if remove_self and 'self' in valid_keys:
        valid_keys.remove('self')

    func_dna = {k: point[k] for k in point if k in valid_keys} # filter to get only params accepted by func

    return func_dna
