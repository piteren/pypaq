from copy import deepcopy
import math
import random
from typing import Optional, Dict, Tuple

from pypaq.pms.base import P_VAL, POINT, PSDD, PMSException

NO_REF = '<<< NO-REF >>>' # axis value when no reference value given (since 'None' cannot be used)


# Parameters Space
class PaSpa:
    """
    PaSpa - Parameters Space

    PaSpa is build from PSDD (Params Space Definition Dictionary),
    PaSpa has two properties that may describe its complexity:
    - dim  - dimensionality (= number of axes)
    - rdim - reduced dimensionality (rdim<=dim - since some axes are simpler(tuples, lists of ints))

    Each axis of PaSpa has:
    - type (list, tuple) of (float, int, diff)
    - range (width)

    PaSpa is a metric space, supports L2 distance calculations, normalized to 1 (the longest space diagonal)
    """

    def __init__(self, psdd:PSDD, seed:Optional[int]=None, logger=None):

        self.logger = logger

        self._psdd = psdd
        self.axes = sorted(list(self._psdd.keys()))
        if self.logger:
            self.logger.info(f'*** PaSpa *** inits..')
            self.logger.info(f'> dim: {self.dim}')

        if seed is not None:
            random.seed(seed)

        self._axT, self._axW = self.__axes_type_width()

        if self.logger: self.logger.info(f'> rdim: {self.rdim:.1f}')

        # width of str(value) for axes, used for nice str formatting
        self._str_width = {}
        for axis in self.axes:
            width = max([len(str(e)) for e in self._psdd[axis]])
            if 'list_float' in self._axT[axis]: width = 7
            self._str_width[axis] = width

    ### *************************************************************************** axes / value_on_axis related methods

    # prepares axes type and width + some safety checks
    def __axes_type_width(self) -> Tuple[Dict,Dict]:
        """
        possible axes types:
        - list_float
        - list_int
        - tuple_float
        - tuple_int
        - tuple_diff
        """

        axT: Dict[str, str] = {}   # axis type
        axW: Dict[str, float] = {} # axis width, range (min <-> max), for diff_tuple - number of elements

        for axis in self.axes:

            if not type(self._psdd[axis]) in (list, tuple): self._psdd[axis] = (self._psdd[axis],) # replace P_VAL by (P_VAL,)

            pdef = self._psdd[axis]
            tp = 'tuple' if type(pdef) is tuple else 'list'
            if tp == 'list' and len(set(pdef))!= 2:
                raise PMSException(f'parameter definition with list should have two different elements: >{pdef}<!')

            tpn = 'int' # default
            are_flt = False
            are_dif = False
            for el in pdef:
                if type(el) is float: are_flt = True
                if type(el) is not int and type(el) is not float: are_dif = True
            if are_flt: tpn = 'float'  # downgrade
            if are_dif: tpn = 'diff'   # downgrade

            if tp=='list' and tpn=='diff':
                raise PMSException(f'ERR: axis {axis} defined with list may contain only floats or ints')

            # sort numeric
            if tpn != 'diff':
                pdef = sorted(list(pdef))
                self._psdd[axis] = pdef if tp == 'list' else tuple(pdef) # rollback type of sorted

            axT[axis] = f'{tp}_{tpn}'
            axW[axis] = pdef[-1] - pdef[0] if tpn != 'diff' else len(pdef) - 1

        return axT, axW

    # select value from axis closest to given val
    def __closest(
            self,
            ref_val: float or int,      # for diff_tuple it is index of value in self.__psdd[axis]
            axis: str,
    ) -> float or int:

        val = ref_val  # for list_float type

        axT = self._axT[axis]

        if axT == 'list_int':
            val = int(round(val))

        if 'tuple' in axT:

            # num tuple
            if 'diff' not in axT:
                axis_dst = [(e, abs(e - ref_val)) for e in self._psdd[axis]]
                axis_dst.sort(key=lambda x: x[1])
                val = axis_dst[0][0]

            # diff_tuple - value of the closest index
            else: val = self._psdd[axis][int(round(val))]

        if not self.__value_in_axis(value=val, axis=axis):
            raise PMSException(f'something went wrong since val {val} not in {axis}')

        return val

    # applies noise to given ref_val (samples new value from sub-range defined by noise scale)
    @staticmethod
    def __apply_noise(
            ref_val: float or int,  # reference value (in range og L & R)
            noise_scale: float,     # <0.0;1.0> distance from ref_val as a factor of range where new val will be sampled
            rngL: float or int,     # range Left value
            rngR: float or int,     # range Right value
    ) -> float:

        dist = noise_scale * (rngR-rngL)

        # left side of ref_val
        if random.random() < 0.5:
            rng = ref_val - dist        # new left range
            if rng < rngL: rng = rngL   # limit to given range

        # right size
        else:
            rng = ref_val + dist        # new right range
            if rng > rngR: rng = rngR   # limit to given range

        return random.uniform(ref_val, rng)

    # samples random value from whole axis
    def __random_value_noref(self, axis:str) -> P_VAL:

        axT = self._axT[axis]
        psdd = self._psdd[axis]

        if 'list' in axT:
            a = psdd[0]
            b = psdd[1]
            val = random.uniform(a,b)
            if 'int' in axT: val = int(round(val))
        else: val = random.choice(psdd)

        return val

    # gets random value for axis (algorithm ensures equal probability for both sides of ref_val)
    def __random_value(
            self,
            axis: str,
            ref_val: P_VAL,         # reference value on axis, possible NO_REF
            prob_noise: float,      # probability of shifting value with noise (for list or tuple_with_num)
            noise_scale: float,     # max noise scale (axis width, for list or tuple_with_num)
            prob_axis: float,       # probability of value replacement by one sampled from whole axis (for list or num_tuple)
            prob_diff_axis:float    # probability of value replacement by one sampled from whole axis (for diff_tuple)
    ) -> P_VAL:

        if ref_val == NO_REF:
            return self.__random_value_noref(axis)

        axT = self._axT[axis]
        axis_psd = self._psdd[axis]

        # list or num_tuple
        if 'diff' not in axT:

            # whole axis
            if random.random() < prob_axis:
                return self.__random_value_noref(axis)

            val = ref_val
            # add noise
            if noise_scale and random.random() < prob_noise:
                val = PaSpa.__apply_noise(
                    ref_val=        ref_val,
                    noise_scale=    noise_scale,
                    rngL=           axis_psd[0],
                    rngR=           axis_psd[-1])
                val = self.__closest(val, axis)

        # tuple_diff
        else:

            # whole axis
            if random.random() < prob_diff_axis:
                return self.__random_value_noref(axis)

            val = ref_val
            # add noise (on indexes)
            if noise_scale and random.random() < prob_noise:
                val_ix = axis_psd.index(val)
                val_ix = PaSpa.__apply_noise(
                        ref_val=        val_ix,
                        noise_scale=    noise_scale,
                        rngL=           0,
                        rngR=           len(axis_psd)-1)
                val = self.__closest(val_ix, axis)

        return val

    # checks if given value belongs to an axis of this space
    def __value_in_axis(self, value:P_VAL, axis:str) -> bool:
        if axis not in self._axT:                                      return False # axis not in a space
        if 'list' in self._axT[axis]:
            if type(value) is float and 'int' in self._axT[axis]:      return False # type mismatch
            if self._psdd[axis][0] <= value <= self._psdd[axis][1]:   return True
            else:                                                       return False # value not in a range
        elif value not in self._psdd[axis]:                            return False # value not in a tuple
        return True


    ### *************************************************************************** "point in the space" related methods

    # samples (random) point from whole space or from the surroundings of ref_point
    def sample_point(
            self,
            ref_point: Optional[POINT]= None,   # reference point
            prob_noise=                 0.3,    # probability of shifting value with noise (for list or num_tuple)
            noise_scale=                0.1,    # max noise scale (axis width)
            prob_axis=                  0.1,    # probability of value replacement by one sampled from whole axis (for list or num_tuple)
            prob_diff_axis=             0.3     # probability of value replacement by one sampled from whole axis (for diff_tuple)
    ) -> POINT:
        if ref_point is None:
            ref_point = {axis: NO_REF for axis in self.axes}
        return {axis: self.__random_value(
            axis=           axis,
            ref_val=        ref_point[axis],
            prob_noise=     prob_noise,
            noise_scale=    noise_scale,
            prob_axis=      prob_axis,
            prob_diff_axis= prob_diff_axis) for axis in self.axes}

    # samples GX point from given none, one or two
    def sample_point_GX(
            self,
            point_main: Optional[POINT]=    None,   # main parent, when not given > samples from the whole space
            point_scnd: Optional[POINT]=    None,   # scnd parent, when not given > samples from the surroundings of point_main
            prob_mix=                       0.5,    # probability of mixing two values (for list or num_tuple - but only when both parents are given)
            prob_noise=                     0.3,    # probability of shifting value with noise (for list or num_tuple)
            noise_scale=                    0.2,    # max noise scale (axis width)
            prob_axis=                      0.1,    # probability of value replacement by one sampled from whole axis (for list or num_tuple)
            prob_diff_axis=                 0.3     # probability of value replacement by one sampled from whole axis (for diff_tuple)
    ) -> POINT:

        ref_point = None # when point_main nor point_scnd are given

        if point_main:
            if not self.is_from_space(point_main):
                raise PMSException(f'ERR: point_main: {point_main} not from space: {self._psdd}')

            ref_point = point_main  # select main parent

            # build ref_point with mix or select
            if point_scnd:
                if not self.is_from_space(point_scnd):
                    raise PMSException(f'point_scnd: {point_scnd} not from space: {self._psdd}')

                ref_point = {}
                ratio = 0.5 + random.random() / 2  # mix/select ratio of parent_main, rest from parent_scnd (at least half from parent_main)

                for axis in self.axes:

                    # mix
                    if random.random() < prob_mix:

                        # num
                        if 'diff' not in self._axT[axis]:
                            val = point_main[axis] * ratio + point_scnd[axis] * (1-ratio)
                            val = self.__closest(val, axis)
                        # diff_tuple - mix on indexes
                        else:
                            pm_ix = self._psdd[axis].index(point_main[axis])
                            ps_ix = self._psdd[axis].index(point_scnd[axis])
                            val_ix = pm_ix * ratio + ps_ix * (1-ratio)
                            val = self.__closest(val_ix, axis)
                    
                    # select
                    else: val = point_main[axis] if random.random() < ratio else point_scnd[axis]

                    ref_point[axis] = val

        # sample from the surroundings of ref_point
        return self.sample_point(
            ref_point=      ref_point,
            prob_noise=     prob_noise,
            noise_scale=    noise_scale,
            prob_axis=      prob_axis,
            prob_diff_axis= prob_diff_axis)

    # samples 2 corner points with max distance (1 in normalized space)
    def sample_corners(self) -> (POINT, POINT):
        pa = {}
        pb = {}
        left = [0 if random.random()>0.5 else 1 for _ in range(self.dim)] # left/right
        for aIX in range(self.dim):
            ax = self.axes[aIX]
            vl = self._psdd[ax][0]
            vr = self._psdd[ax][-1]
            pa[ax] = vl
            pb[ax] = vr
            if left[aIX]:
                pa[ax] = vr
                pb[ax] = vl
        return pa, pb

    # checks if given point comes from this space
    def is_from_space(self, point:POINT) -> bool:
        if set(point.keys()) != set(self.axes): return False
        for axis in point:
            if not self.__value_in_axis(value=point[axis], axis=axis):
                return False
        return True

    # distance (L2, in normalized space) between two points in this space
    def distance(self, pa:POINT, pb:POINT) -> float:
        dist_pow_sum = 0
        for axis in pa:
            if self._axW[axis] > 0:
                dist = self._psdd[axis].index(pa[axis]) - self._psdd[axis].index(pb[axis]) \
                    if 'diff' in self._axT[axis] else \
                    pa[axis] - pb[axis]
                dist_pow_sum += (dist / self._axW[axis]) ** 2
        return  math.sqrt(dist_pow_sum) / math.sqrt(self.dim)

    # prepares normalized point of p, where value of each axis is represented as a float <0.0;1.0>
    def point_normalized(self, p:POINT) -> POINT:

        pn: Dict[str,float] = {}

        for axis in p:

            if self._axT[axis] == 'list_float':
                pn[axis] = (p[axis] - self._psdd[axis][0]) / self._axW[axis] # position in float range

            else:

                # |_._|_._|_._| <- each point is placed in the middle of _ _ (sub width)
                if self._axT[axis] == 'list_int':
                    num_elements = self._axW[axis] + 1
                    v_ix = p[axis] - self._psdd[axis][0]
                else:
                    num_elements = len(self._psdd[axis])
                    v_ix = self._psdd[axis].index(p[axis])

                sw = 1 / num_elements # sub width
                pn[axis] = sw/2 + v_ix*sw # half of sw + 1/2 sw

        return pn

    ### *********************************************************************************** space methods and properties

    # merges two PSDD
    @staticmethod
    def merge_psdd(
            psdd_a: PSDD,
            psdd_b: PSDD) -> PSDD:
        return (PaSpa(psdd_a) + PaSpa(psdd_b)).psdd

    # returns copy of self._psdd
    @property
    def psdd(self) -> PSDD:
        return deepcopy(self._psdd)

    @property
    def dim(self) -> int:
        return len(self._psdd)

    # calculates reduced dimensionality of PaSpa
    @property
    def rdim(self) -> float:
        """
        rdim = log10(∏ sq if sq<10 else 10) for all axes
            sq = 10 for list of floats (axis)
            sq = sqrt(len(axis_elements)) for tuple or list of int
        """
        axd = []
        for axis in self.axes:
            axt = self._axT[axis]
            if 'list' in axt: sq = 10 if 'float' in axt else math.sqrt(self._axW[axis])
            else:             sq = math.sqrt(len(self._psdd[axis]))
            axd.append(sq if sq<10 else 10)
        mul = 1
        for e in axd: mul *= e
        return math.log10(mul)

    # returns number of points for finite number or None
    @property
    def n_points(self) -> Optional[int]:
        n = 1
        for axis in self._axT:
            if self._axT[axis] == 'list_float':
                return None
            else:
                if self._axT[axis] == 'list_int':
                    n *= self._axW[axis] + 1
                else:
                    n *= len(self._psdd[axis])
        return n

    # same axes, same definitions, same L
    def __eq__(self, other):
        if self.axes != other.axes: return False
        for k in self._psdd.keys():
            if self._psdd[k] != other._psdd[k]:
                return False
        return True

    # adds two PaSpa
    def __add__(self, other:"PaSpa") -> "PaSpa":
        psdd_a = self.psdd
        psdd_b = other.psdd
        psdd_merged = {}
        psdd_merged.update(psdd_a)
        for ax in other.axes:
            if ax not in psdd_merged:
                psdd_merged[ax] = psdd_b[ax]
            else:
                if not type(psdd_merged[ax]) == type(psdd_b[ax]):
                    raise PMSException(f'ERR: types of axes \'{ax}\' in both PaSpa differs')
                if 'tuple' in other._axT[ax]:
                    # add new elements to the end
                    if 'diff' in other._axT[ax]:
                        merged = list(psdd_merged[ax])
                        for e in psdd_b[ax]:
                            if e not in merged: merged.append(e)
                    else:
                        merged = list(set(list(psdd_merged[ax]) + list(psdd_b[ax])))
                        merged.sort()
                    psdd_merged[ax] = tuple(merged)
                else:
                    ranges = psdd_merged[ax] + psdd_b[ax]
                    psdd_merged[ax] = [min(ranges), max(ranges)]
        return PaSpa(psdd_merged)

    # returns info(string) about self
    def __str__(self):
        info = f'*** PaSpa *** (dim: {self.dim}, rdim: {self.rdim:.1f}) parameters space:\n'
        max_ax_l = 0
        max_ps_l = 0
        for axis in self.axes:
            if len(axis)                > max_ax_l: max_ax_l = len(axis)
            if len(str(self._psdd[axis])) > max_ps_l: max_ps_l = len(str(self._psdd[axis]))
        if max_ax_l > 40: max_ax_l = 40
        if max_ps_l > 70: max_ps_l = 70

        for axis in self.axes:
            info += f'> {axis:{max_ax_l}s}  {str(self._psdd[axis]):{max_ps_l}s}  {self._axT[axis]:11s}  width: {self._axW[axis]}\n'

        if self.n_points:
            info += f'number of points in space: {self.n_points}\n'

        return info[:-1]