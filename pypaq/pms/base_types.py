"""

 2018 (c) piteren

    PSDD - Parameters Space Definition Dict
        Dictionary that defines space of parameters, is used as an "simple" representation of space by some classes

        PSDD - {axis(parameter name): list or tuple or value}
            > list of ints or floats defines continuous range
            > tuple may contain elements of any type (even non-numeric), may have one
            > any other type is considered to be a constant (single value)

            example:
            {   'a':    [0.0, 1],               # range of floats
                'b':    (-1,-7,10,15.5,90,30),  # set of num(float) values, num will be sorted
                'c':    ('tat','mam','kot'),    # set of diff values
                'd':    [0,10],                 # range of ints
                'e':    (-2.0,2,None)}          # set of diff values
                'f':    (16.2,)}                # single value

"""

from typing import Any, Dict, List, Tuple

from pypaq.lipytools.little_methods import float_to_str

AXIS =  str                                     # axis type (parameter name)
P_VAL = float or int or Any                     # point value (parameter value)
POINT = Dict[AXIS, P_VAL]                       # point (dictionary {parameter: value}) ~ DNA

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

# TODO: put here functions that (check, validate, extract info from) PSDD