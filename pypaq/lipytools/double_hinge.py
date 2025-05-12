from pypaq.exception import PyPaqException


# double hinge function _/**
def double_hinge(
        a_value: float,
        b_value: float,
        a_point: float,
        b_point: float,
        point, # int, float, np.ndarray
) -> float:
    """ returns:
    - a_value for point <= a_point
    - b_value for point >= b_point
    - linear interpolation from a_value to b_value in range (a_point;b_point) """

    if b_point < a_point or b_point == a_point and a_value != b_value:
        raise PyPaqException('bad arguments!')

    if point <= a_point:
        return a_value
    if point >= b_point:
        return b_value
    y = (point - a_point) / (b_point - a_point)
    return a_value + (b_value - a_value) * y