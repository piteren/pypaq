from pypaq.exception import PyPaqException


# double hinge function _/**
def double_hinge(
        s_val: float,
        e_val: float,
        a: int,
        b: int,
        counter: int,
) -> float:
    """ returns:
    - s_val for counter <= a
    - linear interpolation from s_val to e_val in range (a;b)
    - e_val for counter >= b """

    if b < a or b == a and s_val != e_val:
        raise PyPaqException('bad arguments!')

    if counter <= a:
        return s_val
    if counter >= b:
        return e_val
    y = (counter - a) / (b - a)
    return s_val + (e_val-s_val) * y