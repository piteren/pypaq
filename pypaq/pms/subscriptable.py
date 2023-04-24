from copy import deepcopy
from typing import List, Optional, Union, Dict, Set, Tuple

from pypaq.textools.text_metrics import lev_dist
from pypaq.pms.base import POINT, PSDD, PMSException
from pypaq.pms.paspa import PaSpa


# returns nice str information about dict differences: db against da
def dict_diff(da:Dict, db:Dict) -> str:

    nfo = ''

    missing_keys = []
    for k in da:
        if k not in db:
            missing_keys.append(k)

    if missing_keys:
        nfo += f'missing keys: {missing_keys}\n'

    new_keys = []
    new_values = []
    for k in db:
        if k not in da:
            new_keys.append(k)
        else:
            if da[k] != db[k]:
                new_values.append(k)

    if new_keys:
        nfo += f'new keys: {new_keys}\n'
        for k in new_keys:
            nfo += f' > {k}: {db[k]}\n'

    if new_values:
        nfo += f'new values: {new_values}\n'
        for k in new_values:
            nfo += f' > {k}: {db[k]}\n'

    if nfo: nfo = nfo[:-1]
    return nfo


# implements dict-like style access to its (self) parameters
class Subscriptable:
    """
    Subscriptable
        its (object) parameters (POINT) may be accessed in dict-like style [], but are IS NOT a dict type
        protected fields may also be accessed with [], but rather should not
    """

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in vars(self)

    # ********************************************************************************************************** getters

    # returns list of all object fields
    def get_all_fields(self, protected=False) -> List[str]:
        fields = [pm for pm in vars(self) if not pm.startswith('_Subscriptable__')]
        if not protected:
            fields = [pm for pm in fields if not pm.startswith('_')]
        return fields

    # prepares list of params managed by Subscriptable (all but private and protected)
    def get_managed_params(self) -> List[str]:
        return self.get_all_fields(protected=False)

    # returns dict of params managed by Subscriptable (POINT of ALL object fields)
    def get_point(self) -> POINT:
        return {k: self[k] for k in self.get_managed_params()}

    # *************************************************************************************************** update & check

    # update in dict-like style
    def update(self, dct:dict) -> None:
        for key in dct:
            self.__setitem__(key, dct[key])

    # checks for self.params + params for similarity, returns True if already got similar
    def check_params_sim(
            self,
            params: Optional[Union[Dict,List]]= None,
            lev_dist_diff: int=                 1
    ) -> Optional[Set[Tuple[str,str]]]:

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

        return found or None


    def __str__(self):
        s = f'{self.__class__.__name__} (Subscriptable):\n'
        pms = self.get_point()
        if pms:
            for k in sorted(pms.keys()):
                s += f' > {k:30s}: {pms[k]}\n'
        else:
            s += f'---empty---\n'
        return s[:-1]


# extends Subscriptable with GX on POINT
class SubGX(Subscriptable):
    """
    SubGX - Subscriptable with GX (genetic crossing algorithm)
        implements GX on parents POINTs
        GX algorithm builds child POINT from:
            - parameters of parents POINTs
            - and parameters sampled from merged both parents PSDD
        it is also possible to preform GX of SINGLE parent
    """

    def __init__(
            self,
            name: Optional[str]=    None,
            family: Optional[str]=  None, # family of GXable
            psdd: Optional[PSDD]=   None, # PSDD of GXable
            **kwargs):

        if not name:
            name = self.__class__.__name__
        self.name = name

        self.family = family
        # INFO: all keys of self.psdd should be present in self
        #  - it is not checked now (while init) since self may be updated even after init, IT IS checked while GX
        self.psdd = psdd or {}

        # sample kwargs from psdd
        if not kwargs:

            if not psdd:
                raise PMSException('when SubGX \'kwargs\' not given, \'psdd\' must be given')

            paspa = PaSpa(psdd=psdd)
            kwargs = paspa.sample_point_GX()

        self.update(kwargs)

    # checks for compatibility of families, True when are equal or when at least one family is None
    @staticmethod
    def families_compatible(
            parent_main: "SubGX",
            parent_scnd: Optional["SubGX"]= None
    ) -> bool:

        point_main = parent_main.get_point()
        point_scnd = parent_scnd.get_point() if parent_scnd else None

        if point_scnd is None: return True

        if not point_scnd: point_scnd = {}
        fmm = point_main.get('family', None)
        fms = point_scnd.get('family', None)

        if fmm is not None and fms is not None and fmm != fms: return False

        return True

    # GX on SubGX POINTs
    @staticmethod
    def gx_point(
            parent_main: "SubGX",
            parent_scnd: Optional["SubGX"]= None,
            name_child: Optional[str]=      None,
            prob_mix=                       0.5,
            prob_noise=                     0.3,
            noise_scale=                    0.1,
            prob_axis=                      0.1,
            prob_diff_axis=                 0.3
    ) -> POINT:

        if not SubGX.families_compatible(parent_main, parent_scnd):
            raise PMSException('ERR: parents families not compatible')

        point_main = parent_main.get_point()
        point_scnd = parent_scnd.get_point() if parent_scnd else None

        psdd_main = point_main.get('psdd', {})
        psdd_scnd = point_scnd.get('psdd', {}) if point_scnd is not None else None
        psdd_merged = PaSpa.merge_psdd(
                psdd_a= psdd_main,
                psdd_b= psdd_scnd) if psdd_scnd is not None else psdd_main

        paspa_merged = PaSpa(psdd=psdd_merged)

        paspa_axes_not_in_parent = [a for a in paspa_merged.axes if a not in point_main]
        if paspa_axes_not_in_parent:
            raise PMSException(f'ERR: paspa axes not in parent_main: {paspa_axes_not_in_parent}')

        # sub-points limited to axes of psdd_merged (PaSpa.sample_point_GX() does not accept other axes..)
        point_main_psdd = {k: point_main[k] for k in psdd_merged}
        point_scnd_psdd = {k: point_scnd[k] for k in psdd_merged} if point_scnd is not None else None

        point_gx = paspa_merged.sample_point_GX(
            point_main=     point_main_psdd,
            point_scnd=     point_scnd_psdd,
            prob_mix=       prob_mix,
            prob_noise=     prob_noise,
            noise_scale=    noise_scale,
            prob_axis=      prob_axis,
            prob_diff_axis= prob_diff_axis)

        # point_child is based on parent_main with updated values of point_gx
        point_child: POINT = deepcopy(point_main)
        point_child.update(point_gx)
        point_child['psdd'] = psdd_merged # update psdd to psdd_merged

        name_merged = f'{parent_main.name}+{parent_scnd.name}_(gxp)' if parent_scnd else f'{parent_main.name}_(sgxp)'
        point_child['name'] = name_child or name_merged

        return point_child

    # gxable_point is a self POINT limited to axes of self.psdd
    @property
    def gxable_point(self) -> POINT:
        return {k: self[k] for k in self.psdd}


    def __str__(self):
        s = f'{self.__class__.__name__} (SubGX) name: {self.name}, family: {self.family}\n'
        pms = self.get_point()
        for k in sorted(pms.keys()):
            p = f'param: {k}'
            v = f'value: {pms[k]}'
            r = f'ranges:{self.psdd[k]}' if k in self.psdd else ''
            s += f'{p:30s} {v:30s} {r}\n'
        return s[:-1]