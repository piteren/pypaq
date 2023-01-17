"""

 2022 (c) piteren

    Subscriptable - class of objects which parameters may be accessed in dict-like style [], but are not dict type
        - protected fields may also be accessed with [], but rather should not

    SubGX

        Class of objects that may perform GX on DNA

        DNA is a POINT in context of class __init__, type(DNA) == type(POINT),
        it is a complete set of kwargs needed to perform init of a class instance.

        DNA of child comes from parent fields and those sampled with GX algorithm.

        GX of one parent is a procedure of sampling child from the space of parent with GXA parameters.
"""

from copy import deepcopy
from typing import List, Optional, Union, Dict, Set, Tuple

from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.textools.text_metrics import lev_dist
from pypaq.pms.base_types import POINT, PSDD
from pypaq.pms.paspa import PaSpa


class Subscriptable:

    def __init__(self, logger=None):
        self.__log = logger or get_pylogger()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in vars(self)

    # ********************************************************************************************************** getters

    # returns list of all object fields
    def get_all_fields(self, protected=False) -> List[str]:
        fields = [pm for pm in vars(self)]
        if not protected: fields = [f for f in fields if not f.startswith('_')]
        return fields

    # prepares list of params managed by Subscriptable (all but private and protected)
    def get_managed_params(self) -> List[str]:
        return self.get_all_fields(protected=False)

    # returns dict of params managed by Subscriptable (POINT of ALL object fields)
    def get_point(self) -> POINT:
        return {k: self[k] for k in self.get_managed_params()}

    # *************************************************************************************************** update & check

    # update in dict-like style
    def update(self, dct:dict):
        for key in dct:
            self.__setitem__(key, dct[key])

    # checks for self.params + params for similarity, returns True if already got similar
    def check_params_sim(
            self,
            params: Optional[Union[Dict,List]]= None,
            lev_dist_diff: int=                 1) -> Optional[Set[Tuple[str,str]]]:

        if params is None: params = []
        paramsL = params if type(params) is list else list(params.keys())
        self_paramsL = list(self.get_point().keys())
        diff_paramsL = [par for par in paramsL if par not in self_paramsL]

        all_params = sorted(list(set([p.lower() for p in self_paramsL + diff_paramsL])))

        #found_any = False
        found = set()
        for pa in all_params:
            for pb in all_params:
                if pa != pb:
                    levD = lev_dist(pa,pb)
                    if levD <= lev_dist_diff:
                        found.add(tuple(sorted([pa, pb])))

        if found:
            self.__log.warning('Subscriptable was asked to check for params similarity and found:')
            for pa,pb in found: self.__log.warning(f'> params \'{pa}\' and \'{pb}\' are too CLOSE !!!')

        return found or None

    def __str__(self):
        s = f'\n(Subscriptable):\n'
        pms = self.get_point()
        for k in sorted(pms.keys()):
            p = f'param: {k}'
            v = f'value: {pms[k]}'
            s += f'{p:30s} {v:30s}\n'
        return s[:-1]

# adds GX on DNA to Subscriptable
class SubGX(Subscriptable):

    def __init__(
            self,
            name: str,
            family: Optional[str]=  None, # family of GXable
            psdd: Optional[PSDD]=   None, # PSDD of GXable
            logger=                 None,
            **kwargs):

        self.__log = logger or get_pylogger()

        Subscriptable.__init__(self, logger=get_hi_child(self.__log))

        self.name = name
        self.family = family
        self.psdd = psdd or {}
        self.update(kwargs)
        # INFO: all keys of self.psdd should be present in self
        #  - it is not checked now (while init) since self may be updated even after init, IT IS checked while GX
        self.__log.info(f'*** Subscriptable *** name: {self.name} initialized, family: {self.family}, psdd: {self.psdd}')

    # returns self POINT limited to axes included in self.psdd
    def get_gxable_point(self) -> POINT:
        point_gxable = {k: self[k] for k in self.psdd}
        return point_gxable

    # checks for compatibility of families stored in given DNA, True when are equal or when at least one family is None
    @staticmethod
    def families_compatible(
            parent_main: "SubGX",
            parent_scnd: Optional["SubGX"]= None) -> bool:

        dna_main = parent_main.get_point()
        dna_scnd = parent_scnd.get_point() if parent_scnd else None

        if dna_scnd is None: return True

        if not dna_scnd: dna_scnd = {}
        fmm = dna_main.get('family', None)
        fms = dna_scnd.get('family', None)
        if fmm is not None and fms is not None and fmm != fms: return False
        return True

    # GX on DNA (POINTs)
    @staticmethod
    def gx_dna(
            parent_main: "SubGX",
            parent_scnd: Optional["SubGX"]= None,
            name_child: Optional[str]=      None,
            prob_mix=                       0.5,
            prob_noise=                     0.3,
            noise_scale=                    0.1,
            prob_axis=                      0.1,
            prob_diff_axis=                 0.3) -> POINT:

        assert SubGX.families_compatible(parent_main, parent_scnd), 'ERR: families not compatible'

        dna_main = parent_main.get_point()
        dna_scnd = parent_scnd.get_point() if parent_scnd else None

        psdd_main = dna_main.get('psdd', {})
        psdd_scnd = dna_scnd.get('psdd', {}) if dna_scnd is not None else None
        psdd_merged = PaSpa.merge_psdd(
                psdd_a= psdd_main,
                psdd_b= psdd_scnd) if psdd_scnd is not None else psdd_main

        paspa_merged = PaSpa(psdd=psdd_merged)

        paspa_axes_not_in_parent = [a for a in paspa_merged.axes if a not in dna_main]
        assert not paspa_axes_not_in_parent, f'ERR: paspa axes not in parent_main: {paspa_axes_not_in_parent}'

        # sub-points limited to axes of psdd_merged (PaSpa.sample_point_GX() does not accept other axes..)
        dna_main_psdd = {k: dna_main[k] for k in psdd_merged}
        dna_scnd_psdd = {k: dna_scnd[k] for k in psdd_merged} if dna_scnd is not None else None

        point_gx = paspa_merged.sample_point_GX(
            point_main=     dna_main_psdd,
            point_scnd=     dna_scnd_psdd,
            prob_mix=       prob_mix,
            prob_noise=     prob_noise,
            noise_scale=    noise_scale,
            prob_axis=      prob_axis,
            prob_diff_axis= prob_diff_axis)

        # dna_child is based on parent_main with updated values of point_gx
        dna_child: POINT = deepcopy(dna_main)
        dna_child.update(point_gx)
        dna_child['psdd'] = psdd_merged # update psdd to psdd_merged

        name_merged = f'{parent_main.name}+{parent_scnd.name}_(gxp)' if parent_scnd else f'{parent_main.name}_(sgxp)'
        dna_child['name'] = name_child or name_merged

        return dna_child

    def __str__(self):
        s = f'\n(Subscriptable) name: {self.name}, family: {self.family}\n'
        pms = self.get_point()
        for k in sorted(pms.keys()):
            p = f'param: {k}'
            v = f'value: {pms[k]}'
            r = f'ranges:{self.psdd[k]}' if k in self.psdd else ''
            s += f'{p:30s} {v:30s} {r}\n'
        return s[:-1]