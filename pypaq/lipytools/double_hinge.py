import numpy as np

from pypaq.exception import PyPaqException


def _double_hinge_np(
        a_value: float,
        b_value: float,
        a_step: float|int,
        b_step: float|int,
        step: np.ndarray,
        smooth: bool = False,
) -> np.ndarray:
    """ double hinge function _/**
    returns:
    - a_value for step <= a_step
    - b_value for step >= b_step
    - linear interpolation from a_value to b_value in range (a_step;b_step)
    or cosine (ease in-out) interpolation when smooth """

    if b_step < a_step or b_step == a_step and a_value != b_value:
        raise PyPaqException('wrong arguments values!')

    x = (step - a_step) / (b_step - a_step) # position on x-axis
    x = np.maximum(0.0, np.minimum(1.0, x)) # trim to <0.0;1.0>
    if smooth:
        x = 0.5 - 0.5 * np.cos(np.pi * x)  # slow start, fast middle, slow end
    return a_value + (b_value - a_value) * x


def double_hinge(
        a_value: float,
        b_value: float,
        a_step: float,
        b_step: float,
        step: int|float|np.ndarray,
        smooth: bool = False,
) -> float|np.ndarray:
    r = _double_hinge_np(a_value, b_value, a_step, b_step, step, smooth)
    if type(step) is not np.ndarray:
        r = float(r)
    return r