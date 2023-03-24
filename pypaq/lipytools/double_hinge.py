from pypaq.exception import PyPaqException


# double hinge function _/**
def double_hinge(
        s_val: float,
        e_val: float,
        sf: float,
        ef: float,
        counter: int,
        max_count: int,
) -> float:
    """
    returns:
        - s_val in range <0;A> steps
        - linear interpolation from s_val to e_val in range <A;B>
        - e_val in range <A;max_count>

    counter (step) belongs to <0;max_count>
    A = max_count * sf
    B = max_count * (1-ef)
    0 <= A <= B <= max_count
    """
    if sf + ef > 1:
        raise PyPaqException(f'sf + ef cannot be higher than 1')

    x = counter / max_count         # where we are in time (x)
    y = (x-sf)/(1-sf-ef)            # how high we are
    if y<0: y=0
    if y>1: y=1
    val = s_val + (e_val-s_val)*y   # final value
    return val