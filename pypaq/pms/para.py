from copy import deepcopy
from typing import List, Optional, Union, Dict, Set, Tuple

from pypaq.textools.text_metrics import lev_dist
from pypaq.pms.base import POINT, PSDD, PMSException
from pypaq.pms.paspa import PaSpa


def dict_diff(da:Dict, db:Dict) -> str:
    """ returns nice str information about dict differences: db against da """

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


class Para:
    """ Parameters Access
    its parameters (point) may be accessed in dict-like style []
    but Para IS NOT a dict
    protected fields may also be accessed with [], but rather should not """

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return key in vars(self)

    # ********************************************************************************************************** getters

    def get_all_params(self, protected=False) -> List[str]:
        """ returns list of all self parameters """
        fields = [pm for pm in vars(self) if not pm.startswith('_Para__')]
        if not protected:
            fields = [pm for pm in fields if not pm.startswith('_')]
        return fields

    def get_managed_params(self) -> List[str]:
        """ prepares list of parameters managed by Para (all but private and protected) """
        return self.get_all_params(protected=False)

    def get_point(self) -> POINT:
        """ returns POINT of parameters managed by Para """
        return {k: self[k] for k in self.get_managed_params()}

    # *************************************************************************************************** update & check

    def update(self, d:dict) -> None:
        """ update in dict-like style """
        for key in d:
            self.__setitem__(key, d[key])

    def check_params_sim(
            self,
            params: Optional[Union[Dict,List]]= None,
            lev_dist_diff: int=                 1,
    ) -> Optional[Set[Tuple[str,str]]]:
        """ checks self.params + params for similarity against given
        returns True if already got similar """

        if params is None:
            params = []
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
        s = f'{self.__class__.__name__} (Para):\n'
        pms = self.get_point()
        if pms:
            for k in sorted(pms.keys()):
                s += f' > {k:30s}: {pms[k]}\n'
        else:
            s += f'---empty---\n'
        return s[:-1]


class ParaGX(Para):
    """ ParaGX - Para with GX (genetic crossing algorithm)
    implements GX on parents POINTs

    GX algorithm builds child POINT from:
        - parameters of parents POINTs
        - parameters sampled from merged both parents PSDD
    it is also possible to preform GX of SINGLE parent
    """

    def __init__(
            self,
            name: Optional[str]=    None,
            family: Optional[str]=  '__gx-fam__', # family of GXable, here default, for GX functionality family needs to be given
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
                raise PMSException('when ParaGX \'kwargs\' not given, \'psdd\' must be given')

            paspa = PaSpa(psdd=psdd, loglevel=30)
            kwargs = paspa.sample_point_GX()

        self.update(kwargs)

    @staticmethod
    def families_compatible(
            parentA: "ParaGX",
            parentB: Optional["ParaGX"]= None
    ) -> bool:
        """ checks for compatibility of families
        returns True when families match and are not None """

        pointA = parentA.get_point()
        fmA = pointA.get('family', None)
        pointB = parentB.get_point() if parentB else None

        if not pointB:
            return fmA is not None

        else:
            if fmA is None:
                return False

            fmB = pointB.get('family', None)
            return fmA == fmB

    @staticmethod
    def gx_point(
            parentA: "ParaGX",
            parentB: Optional["ParaGX"]=    None,
            name_child: Optional[str]=      None,
            prob_mix=                       0.5,
            prob_noise=                     0.3,
            noise_scale=                    0.1,
            prob_axis=                      0.1,
            prob_diff_axis=                 0.3
    ) -> POINT:
        """ GX on ParaGX POINTs """

        if not ParaGX.families_compatible(parentA, parentB):
            raise PMSException('ERR: parents families not compatible')

        pointA = parentA.get_point()
        pointB = parentB.get_point() if parentB else None

        psddA = pointA.get('psdd', {})
        psddB = pointB.get('psdd', {}) if pointB is not None else None
        psdd_merged = PaSpa.merge_psdd(
                psdd_a= psddA,
                psdd_b= psddB) if psddB is not None else psddA

        paspa_merged = PaSpa(psdd=psdd_merged, loglevel=30)

        paspa_axes_not_in_parent = [a for a in paspa_merged.axes if a not in pointA]
        if paspa_axes_not_in_parent:
            raise PMSException(f'ERR: PaSpa axes not in parentA: {paspa_axes_not_in_parent}')

        # sub-points limited to axes of psdd_merged (PaSpa.sample_point_GX() does not accept other axes..)
        pointA_psdd = {k: pointA[k] for k in psdd_merged}
        pointB_psdd = {k: pointB[k] for k in psdd_merged} if pointB is not None else None

        point_gx = paspa_merged.sample_point_GX(
            pointA=         pointA_psdd,
            pointB=         pointB_psdd,
            prob_mix=       prob_mix,
            prob_noise=     prob_noise,
            noise_scale=    noise_scale,
            prob_axis=      prob_axis,
            prob_diff_axis= prob_diff_axis)

        # point_child is based on parentA with updated values of point_gx
        point_child: POINT = deepcopy(pointA)
        point_child.update(point_gx)
        point_child['psdd'] = psdd_merged # update psdd to psdd_merged

        name_merged = f'{parentA.name}+{parentB.name}_(gxp)' if parentB else f'{parentA.name}_(sgxp)'
        point_child['name'] = name_child or name_merged

        return point_child

    @property
    def gxable_point(self) -> POINT:
        """ gxable_point is a self POINT limited to axes of self.psdd """
        return {k: self[k] for k in self.psdd}


    def __str__(self):
        s = f'{self.__class__.__name__} (ParaGX) name: {self.name}, family: {self.family}\n'
        pms = self.get_point()
        for k in sorted(pms.keys()):
            p = f'param: {k}'
            v = f'value: {pms[k]}'
            r = f'ranges:{self.psdd[k]}' if k in self.psdd else ''
            s += f'{p:30s} {v:30s} {r}\n'
        return s[:-1]