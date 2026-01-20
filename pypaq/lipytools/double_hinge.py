import numpy as np

from pypaq.exception import PyPaqException


def _double_hinge_np(
        a_value: float,
        b_value: float,
        a_point: float,
        b_point: float,
        point: np.ndarray,
) -> np.ndarray:
    """ double hinge function _/**
    returns:
    - a_value for point <= a_point
    - b_value for point >= b_point
    - linear interpolation from a_value to b_value in range (a_point;b_point) """

    if b_point < a_point or b_point == a_point and a_value != b_value:
        raise PyPaqException('wrong arguments values!')

    r = (point - a_point) / (b_point - a_point)
    v_max, v_min = (b_value, a_value) if b_value > a_value else (a_value, b_value)
    r = np.maximum(v_min, np.minimum(v_max, r))
    return a_value + (b_value - a_value) * r


def double_hinge(
        a_value: float,
        b_value: float,
        a_point: float,
        b_point: float,
        point: int|float|np.ndarray,
) -> float|np.ndarray:
    r = _double_hinge_np(a_value, b_value, a_point, b_point, point)
    if type(point) is not np.ndarray:
        r = float(r)
    return r