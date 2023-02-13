"""

 2018 (c) piteren

    PaSpa - Parameters Space

        PaSpa is build from PSDD, it has:
            - dim  - dimensionality (= number of axes)
            - rdim - reduced dimensionality (rdim<=dim - since some axes are simpler(tuples, lists of ints))

        each axis of space has:
            - type (list - continuous, tuple - discrete)
            - range (width)

        PaSpa is a metric space (https://en.wikipedia.org/wiki/Metric_space)
        PaSpa supports L1 and L2 distance calculations, normalized to 1 (1 is the longest diagonal of space)

"""

import math
import random
from typing import Optional

from pypaq.pms.base import P_VAL, POINT, PSDD
from pypaq.lipytools.pylogger import get_pylogger

NO_REF = '__NO-REF__'


# Parameters Space
class PaSpa:

    def __init__(
            self,
            psdd : PSDD,            # params space dictionary
            distance_L2=    True,   # sets L1 or L2 distance for PaSpa
            logger=         None,
            loglevel=       20):

        if not logger:
            logger = get_pylogger(
                name=       'paspa',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.logger = logger

        self.__psdd = psdd
        self.axes = sorted(list(self.__psdd.keys()))
        self.L2 = distance_L2
        self.dim = len(self.__psdd)
        self.logger.info(f'*** PaSpa ***  inits..')
        self.logger.info('> (dim: {self.dim})')

        # resolve axis type and width and some safety checks
        self.__axT = {} # axis type, [list,tuple]_[float,int,diff] list_diff is not allowed
        self.__axW = {} # axis width
        for axis in self.axes:

            if not type(self.__psdd[axis]) in (list,tuple): self.__psdd[axis] = (self.__psdd[axis],) # replace P_VAL by (P_VAL,)

            pdef = self.__psdd[axis]
            tp = 'tuple' if type(pdef) is tuple else 'list'
            assert not (tp == 'list' and len(set(pdef))!= 2), f'ERR: parameter definition with list should have two different elements: >{pdef}<!'

            tpn = 'int' # default
            are_flt = False
            are_dif = False
            for el in pdef:
                if type(el) is float: are_flt = True
                if type(el) is not int and type(el) is not float: are_dif = True
            if are_flt: tpn = 'float'  # downgrade
            if are_dif: tpn = 'diff'   # downgrade

            assert not (tp=='list' and tpn=='diff'), f'ERR: axis {axis} defined with list may contain only floats or ints'

            # sort numeric
            if tpn != 'diff':
                pdef = sorted(list(pdef))
                self.__psdd[axis] = pdef if tp == 'list' else tuple(pdef) # rollback type of sorted

            self.__axT[axis] = f'{tp}_{tpn}' # string like 'list_int'
            self.__axW[axis] = pdef[-1] - pdef[0] if tpn != 'diff' else len(pdef) - 1 # range, for diff_tuple - number of elements

        self.rdim = self.get_rdim()
        self.logger.info(f' > rdim: {self.rdim:.1f}')

        # width of str(value) for axes, used for str formatting
        self.__str_width = {}
        for axis in self.axes:
            width = max([len(str(e)) for e in self.__psdd[axis]])
            if 'list_float' in self.__axT[axis]: width = 7
            self.__str_width[axis] = width

    # returns copy of self.__psdd
    def get_psdd(self) -> PSDD:
        psdd = {}
        psdd.update(self.__psdd)
        return psdd

    # calculates reduced dimensionality of PaSpa
    def get_rdim(self):
        """
        rdim = log10(∏ sq if sq<10 else 10) for all axes
            sq = 10 for list of floats (axis)
            sq = sqrt(len(axis_elements)) for tuple or list of int
        """
        axd = []
        for axis in self.axes:
            axt = self.__axT[axis]
            if 'list' in axt: sq = 10 if 'float' in axt else math.sqrt(self.__axW[axis])
            else:             sq = math.sqrt(len(self.__psdd[axis]))
            axd.append(sq if sq<10 else 10)
        mul = 1
        for e in axd: mul *= e
        return math.log10(mul)

    # select value from axis closest to given val
    def __closest(
            self,
            ref_val: float or int,      # for diff_tuple it is index of value in self.__psdd[axis]
            axis: str) -> float or int:

        val = ref_val  # for list_float type

        axT = self.__axT[axis]

        if axT == 'list_int': val = int(round(val))

        if 'tuple' in axT:

            # num tuple
            if 'diff' not in axT:
                axis_dst = [(e, abs(e - ref_val)) for e in self.__psdd[axis]]
                axis_dst.sort(key=lambda x: x[1])
                val = axis_dst[0][0]
            # diff_tuple - value of closest index
            else: val = self.__psdd[axis][int(round(val))]

        assert self.__value_in_axis(val, axis)
        return val

    # applies noise to given ref_val (samples new value from sub-range defined by noise scale)
    @staticmethod
    def __apply_noise(
            ref_val: float or int,  # reference value (in range og L & R)
            noise_scale: float,     # <0.0;1.0> distance from ref_val as a factor of range where new val will be sampled
            rngL: float or int,     # range Left value
            rngR: float or int      # range Right value
    ) -> float:
        assert rngL <= ref_val <= rngR, f'ERR: ref_val: {ref_val} not in range: [{rngL};{rngR}]' # TODO: safety check, remove later
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
    def __random_value_noref(self, axis: str) -> P_VAL:

        axT = self.__axT[axis]
        psdd = self.__psdd[axis]

        if 'list' in axT:
            a = psdd[0]
            b = psdd[1]
            val = random.uniform(a,b)
            if 'int' in axT: val = int(round(val))
        else: val = random.choice(psdd)

        assert self.__value_in_axis(value=val, axis=axis) # TODO: safety check, remove later
        return val

    # gets random value for axis (algorithm ensures equal probability for both sides of ref_val)
    def __random_value(
            self,
            axis: str,
            ref_val: P_VAL or NO_REF,   # reference value on axis
            prob_noise: float,          # probability of shifting value with noise (for list or tuple_with_num)
            noise_scale: float,         # max noise scale (axis width, for list or tuple_with_num)
            prob_axis: float,           # probability of value replacement by one sampled from whole axis (for list or num_tuple)
            prob_diff_axis:float        # probability of value replacement by one sampled from whole axis (for diff_tuple)
    ) -> P_VAL:

        if ref_val == NO_REF: return self.__random_value_noref(axis)

        axT = self.__axT[axis]
        axis_psd = self.__psdd[axis]

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

        assert self.__value_in_axis(value=val, axis=axis) # TODO: safety check, remove later
        return val

    # samples (random) point from whole space or from the surroundings of ref_point
    def __sample_point(
            self,
            ref_point: POINT or NO_REF,         # reference point
            prob_noise=                 0.3,    # probability of shifting value with noise (for list or num_tuple)
            noise_scale=                0.1,    # max noise scale (axis width)
            prob_axis=                  0.1,    # probability of value replacement by one sampled from whole axis (for list or num_tuple)
            prob_diff_axis=             0.3     # probability of value replacement by one sampled from whole axis (for diff_tuple)
    ) -> POINT:
        if ref_point == NO_REF: ref_point = {axis: NO_REF for axis in self.axes}
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

        ref_point = NO_REF # for NO_REF samples from whole space

        if point_main:
            assert self.point_from_space(point_main), f'ERR: point_main: {point_main} not from space: {self.__psdd}'

            ref_point = point_main  # select main parent

            # build ref_point with mix or select
            if point_scnd:
                assert self.point_from_space(point_scnd), f'ERR: point_scnd: {point_scnd} not from space: {self.__psdd}'

                ref_point = {}
                ratio = 0.5 + random.random() / 2  # mix/select ratio of parent_main, rest from parent_scnd (at least half from parent_main)

                for axis in self.axes:

                    # mix
                    if random.random() < prob_mix:

                        # num
                        if 'diff' not in self.__axT[axis]:
                            val = point_main[axis] * ratio + point_scnd[axis] * (1-ratio)
                            val = self.__closest(val, axis)
                        # diff_tuple - mix on indexes
                        else:
                            pm_ix = self.__psdd[axis].index(point_main[axis])
                            ps_ix = self.__psdd[axis].index(point_scnd[axis])
                            val_ix = pm_ix * ratio + ps_ix * (1-ratio)
                            val = self.__closest(val_ix, axis)
                    
                    # select
                    else: val = point_main[axis] if random.random() < ratio else point_scnd[axis]

                    ref_point[axis] = val

        # sample from the surroundings of ref_point
        return self.__sample_point(
            ref_point=      ref_point,
            prob_noise=     prob_noise,
            noise_scale=    noise_scale,
            prob_axis=      prob_axis,
            prob_diff_axis= prob_diff_axis)

    # samples 2 corner points with max distance
    def sample_corners(self) -> (POINT, POINT):
        pa = {}
        pb = {}
        left = [0 if random.random()>0.5 else 1 for _ in range(self.dim)] # left/right
        for aIX in range(self.dim):
            ax = self.axes[aIX]
            vl = self.__psdd[ax][0]
            vr = self.__psdd[ax][-1]
            pa[ax] = vl
            pb[ax] = vr
            if left[aIX]:
                pa[ax] = vr
                pb[ax] = vl
        return pa, pb

    # checks if given value belongs to an axis of this space
    def __value_in_axis(self, value: P_VAL, axis: str) -> bool:
        if axis not in self.__axT:                                      return False # axis not in a space
        if 'list' in self.__axT[axis]:
            if type(value) is float and 'int' in self.__axT[axis]:      return False # type mismatch
            if self.__psdd[axis][0] <= value <= self.__psdd[axis][1]:   return True
            else:                                                       return False # value not in a range
        elif value not in self.__psdd[axis]:                            return False # value not in a tuple
        return True

    # checks if point comes from a space (is built from same axes and is in space and)
    def point_from_space(self, point: POINT) -> bool:
        if set(point.keys()) != set(self.axes): return False
        for axis in point:
            if not self.__value_in_axis(value=point[axis], axis=axis):
                return False
        return True

    # distance between two points in this space (normalized to 1 - divided by max space distance)
    def distance(self, pa: POINT, pb: POINT) -> float:

        if self.L2:
            dist_pow_sum = 0
            for axis in pa:
                if self.__axW[axis] > 0:
                    dist = self.__psdd[axis].index(pa[axis]) - self.__psdd[axis].index(pb[axis]) \
                        if 'diff' in self.__axT[axis] else \
                        pa[axis] - pb[axis]
                    dist_pow_sum += (dist / self.__axW[axis])**2
            return  math.sqrt(dist_pow_sum) / math.sqrt(self.dim)
        else:
            dist_abs_sum = 0
            for axis in pa:
                if self.__axW[axis] > 0:
                    dist = self.__psdd[axis].index(pa[axis]) - self.__psdd[axis].index(pb[axis]) \
                        if 'diff' in self.__axT[axis] else \
                        pa[axis] - pb[axis]
                    dist_abs_sum += abs(dist) / self.__axW[axis]
            return dist_abs_sum / self.dim

    # merges two PSDD
    @staticmethod
    def merge_psdd(
            psdd_a: PSDD,
            psdd_b: PSDD) -> PSDD:
        return (PaSpa(psdd_a) + PaSpa(psdd_b)).get_psdd()

    # returns info(string) about self
    def __str__(self):
        info = f'*** PaSpa *** (dim: {self.dim}, rdim: {self.rdim:.1f}) parameters space:\n'
        max_ax_l = 0
        max_ps_l = 0
        for axis in self.axes:
            if len(axis)                > max_ax_l: max_ax_l = len(axis)
            if len(str(self.__psdd[axis])) > max_ps_l: max_ps_l = len(str(self.__psdd[axis]))
        if max_ax_l > 40: max_ax_l = 40
        if max_ps_l > 40: max_ps_l = 40

        for axis in self.axes:
            info += f' > {axis:{max_ax_l}s}  {str(self.__psdd[axis]):{max_ps_l}s}  {self.__axT[axis]:11s}  width: {self.__axW[axis]}\n'
        return info[:-1]

    # same axes, same definitions, same L
    def __eq__(self, other):
        if self.axes != other.axes: return False
        for k in self.__psdd.keys():
            if self.__psdd[k] != other.__psdd[k]:
                return False
        if self.L2 != other.L2:
            return False
        return True

    def __add__(self, other: "PaSpa") -> "PaSpa":
        psdd_a = self.get_psdd()
        psdd_b = other.get_psdd()
        psdd_merged = {}
        psdd_merged.update(psdd_a)
        for ax in other.axes:
            if ax not in psdd_merged: psdd_merged[ax] = psdd_b[ax]
            else:
                assert type(psdd_merged[ax]) == type(psdd_b[ax]), f'ERR: types of axes {ax} for PSDD in both PaSpa do not match!'
                if 'tuple' in other.__axT[ax]:
                    # add new elements to the end
                    if 'diff' in other.__axT[ax]:
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