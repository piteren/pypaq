import os
import shutil
from typing import Optional, List

from pypaq.lipytools.printout import stamp
from pypaq.lipytools.files import r_pickle, w_pickle, prep_folder
from pypaq.lipytools.pylogger import get_pylogger, get_child
from pypaq.pms.base import POINT
from pypaq.pms.subscriptable import SubGX


class ParaSaveException(Exception):
    pass


class ParaSave(SubGX):
    """
    Parameters Save
        uses save_topdir where saves/loads POINT (self params)
        implements GX for saved
    """

    PARASAVE_DEFAULTS = {
        'gxable':   True,
        'parents':  [], # list of parents names TODO: what is this for?
    }

    SAVE_TOPDIR = None
    SAVE_FN_PFX = 'point'   # POINT file prefix
    OBJ_SUFFIX = '.dct'     # POINT file suffix of obj(pickle)
    TXT_SUFFIX = '.txt'     # POINT file suffix of text

    def __init__(
            self,
            name:str,
            save_topdir: Optional[str]= None,   # ParaSave top directory, when not given folders functionality (load, save ..) is disabled
            save_fn_pfx: Optional[str]= None,   # ParaSave POINT file prefix
            assert_saved=               False,  # for True asserts that ParaSave has been already saved in save_topdir
            lock_managed_params=        False,  # locks _managed_params to only those known while init
            logger=                     None,
            loglevel=                   20,
            **kwargs):

        self.name = name
        self.save_topdir = save_topdir or self.SAVE_TOPDIR
        self.save_fn_pfx = save_fn_pfx or self.SAVE_FN_PFX

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                add_stamp=  False,
                folder=     ParaSave.__full_dir(name=self.name, save_topdir=self.save_topdir) if self.save_topdir else None,
                level=      loglevel)
        self.__log = logger

        self.__log.info(f'*** ParaSave : {self.name} *** initializes..')
        self.__log.debug(f'> save_topdir: {self.save_topdir}')
        self.__log.debug(f'> save_fn_pfx: {self.save_fn_pfx}')


        if assert_saved and not os.path.isfile(ParaSave.__obj_fn(name, self.save_topdir, self.save_fn_pfx)):
            ex_msg = f'ParaSave {self.name} does not exist!'
            self.__log.error(ex_msg)
            raise ParaSaveException(ex_msg)

        point_saved = ParaSave.load_point(
            name=           self.name,
            save_topdir=    self.save_topdir,
            save_fn_pfx=    self.save_fn_pfx)

        # update in proper order
        self.update(ParaSave.PARASAVE_DEFAULTS)
        self.update(point_saved)
        self.update(kwargs)

        # _managed_params allows to lock managed params only to those given here
        self._managed_params: Optional[List[str]] = None
        if lock_managed_params: self._managed_params = self.get_managed_params()

        point = self.get_point()

        self.__log.debug(f'> ParaSave POINT sources:')
        self.__log.debug(f'>> PARASAVE_DEFAULTS:    {ParaSave.PARASAVE_DEFAULTS}')
        self.__log.debug(f'>> POINT saved:          {point_saved}')
        self.__log.debug(f'>> given kwargs:         {kwargs}')
        self.__log.debug(f'>> managed params:       {self.get_managed_params()}')
        self.__log.debug(f'ParaSave complete POINT: {point}')

        SubGX.__init__(self, logger=get_child(self.__log), **point)


    def get_managed_params(self) -> List[str]:
        if self._managed_params is not None: return self._managed_params
        return SubGX.get_managed_params(self)

    # ************************************************************************************************ folder management

    @staticmethod
    def __full_dir(
            name: str,
            save_topdir: str):
        return f'{save_topdir}/{name}'

    @classmethod
    def __obj_fn(
            cls,
            name: str,
            save_topdir: str,
            save_fn_pfx: str):
        return f'{cls.__full_dir(name, save_topdir)}/{save_fn_pfx}{cls.OBJ_SUFFIX}'

    @classmethod
    def __txt_fn(
            cls,
            name: str,
            save_topdir: str,
            save_fn_pfx: str):
        return f'{cls.__full_dir(name, save_topdir)}/{save_fn_pfx}{cls.TXT_SUFFIX}'

    # loads ParaSave POINT from folder
    @classmethod
    def load_point(
            cls,
            name: str,
            save_topdir: Optional[str]= None,
            save_fn_pfx: Optional[str]= None
    ) -> POINT:
        obj_FN = ParaSave.__obj_fn(
            name=           name,
            save_topdir=    save_topdir or cls.SAVE_TOPDIR,
            save_fn_pfx=    save_fn_pfx or cls.SAVE_FN_PFX)
        if os.path.isfile(obj_FN):
            point: POINT = r_pickle(obj_FN)
            return point
        return {}

    # saves ParaSave POINT to folder (with preview in txt)
    def save_point(self):

        if not self.save_topdir:
            msg = 'cannot save ParaSave, if save directory was not given, aborting'
            self.__log.error(msg)
            raise ParaSaveException(msg)

        obj_FN = ParaSave.__obj_fn(
            name=           self.name,
            save_topdir=    self.save_topdir,
            save_fn_pfx=    self.save_fn_pfx)
        txt_FN = ParaSave.__txt_fn(
            name=           self.name,
            save_topdir=    self.save_topdir,
            save_fn_pfx=    self.save_fn_pfx)

        prep_folder(ParaSave.__full_dir(
            name=           self.name,
            save_topdir=    self.save_topdir))

        if os.path.isfile(obj_FN): shutil.copy(obj_FN, f'{obj_FN}_OLD')
        if os.path.isfile(txt_FN): shutil.copy(txt_FN, f'{txt_FN}_OLD')

        point: POINT = self.get_point()
        w_pickle(point, obj_FN)
        with open(txt_FN, 'w') as file:
            s = f' *** ParaSave POINT saved: {stamp(year=True, letters=None)} ***\n'
            s += ParaSave.dict_2str(point)
            file.write(s)

        self.__log.debug(f'{self.__class__.__name__} {self.name} saved to {self.save_topdir}')

    # loads, next overrides parameters from given kwargs and saves new ParaSave POINT
    @classmethod
    def oversave_point(
            cls,
            name: str,
            save_topdir: Optional[str]= None,
            save_fn_pfx: Optional[str]= None,
            **kwargs):
        psc = ParaSave(
            name=           name,
            save_topdir=    save_topdir or cls.SAVE_TOPDIR,
            save_fn_pfx=    save_fn_pfx or cls.SAVE_FN_PFX,
            loglevel=       30)
        psc.update(kwargs)
        psc.save_point()

    # copies saved ParaSave POINT from one folder to another
    @classmethod
    def copy_saved_point(
            cls,
            name_src: str,
            name_trg: str,
            save_topdir_src: Optional[str]= None,
            save_topdir_trg: Optional[str]= None,
            save_fn_pfx: Optional[str]=     None):

        if not save_topdir_src: save_topdir_src = cls.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX

        ps_src = ParaSave(
            name=           name_src,
            save_topdir=    save_topdir_src,
            save_fn_pfx=    save_fn_pfx,
            assert_saved=   True,
            loglevel=       30)

        ps_trg = ParaSave(
            name=           name_trg,
            save_topdir=    save_topdir_trg or save_topdir_src,
            save_fn_pfx=    save_fn_pfx,
            loglevel=       30)

        point_src = ps_src.get_point()
        for k in ['name','save_topdir']: point_src.pop(k)
        ps_trg.update(point_src)
        ps_trg.save_point()

    # performs GX on saved ParaSave POINT (without even building child objects)
    @classmethod
    def gx_saved_point(
            cls,
            name_parent_main: str,
            name_parent_scnd: Optional[str],                # if not given makes GX only with main parent
            name_child: str,
            save_topdir_parent_main: Optional[str]= None,   # ParaSave top directory
            save_topdir_parent_scnd: Optional[str]= None,   # ParaSave top directory of parent scnd
            save_topdir_child: Optional[str]=       None,   # ParaSave top directory of child
            save_fn_pfx: Optional[str]=             None,   # ParaSave POINT file prefix
    ) -> None:

        if not save_topdir_parent_main: save_topdir_parent_main = cls.SAVE_TOPDIR
        if not save_topdir_parent_scnd: save_topdir_parent_scnd = save_topdir_parent_main
        if not save_topdir_child: save_topdir_child = save_topdir_parent_main
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX

        pm = ParaSave(
            name=           name_parent_main,
            save_topdir=    save_topdir_parent_main,
            save_fn_pfx=    save_fn_pfx,
            assert_saved=   True,
            loglevel=       30)

        ps = ParaSave(
            name=           name_parent_scnd,
            save_topdir=    save_topdir_parent_scnd,
            save_fn_pfx=    save_fn_pfx,
            loglevel=       30) if name_parent_scnd else None

        not_gxable_parents = []
        if not pm['gxable']: not_gxable_parents.append(pm)
        if ps and not ps['gxable']: not_gxable_parents.append(ps)
        if not_gxable_parents:
            raise ParaSaveException('not gxable parents, cannot GX!')
        else:
            # make pm a child and save
            point_child = SubGX.gx_point(
                parent_main=    pm,
                parent_scnd=    ps,
                name_child=     name_child)
            pm.update(point_child)
            pm['_save_topdir'] = save_topdir_child
            pm.parents = [name_parent_main]
            if name_parent_scnd: pm.parents.append(name_parent_scnd)
            pm.save_point()

    # adds gxable check
    @staticmethod
    def gx_point(
            parent_main: "ParaSave",
            parent_scnd: Optional["ParaSave"]=  None,
            name_child: Optional[str]=          None,
            prob_mix=                           0.5,
            prob_noise=                         0.3,
            noise_scale=                        0.1,
            prob_axis=                          0.1,
            prob_diff_axis=                     0.3
    ) -> POINT:
        not_gxable_parents = []
        if not parent_main['gxable']: not_gxable_parents.append(parent_main)
        if parent_scnd and not parent_scnd['gxable']: not_gxable_parents.append(parent_scnd)
        if not_gxable_parents:
            raise ParaSaveException('not gxable parents, cannot GX!')
        else:
            return SubGX.gx_point(
                parent_main=    parent_main,
                parent_scnd=    parent_scnd,
                name_child=     name_child,
                prob_mix=       prob_mix,
                prob_noise=     prob_noise,
                noise_scale=    noise_scale,
                prob_axis=      prob_axis,
                prob_diff_axis= prob_diff_axis)

    # returns nice string of given dict (mostly for .txt preview save)
    @staticmethod
    def dict_2str(d: dict) -> str:
        if d:
            s = ''
            max_len_sk = max([len(k) for k in d.keys()])
            for k, v in sorted(d.items()): s += f'{str(k):{max_len_sk}s} : {str(v)}\n'
            return s[:-1]
        return '--empty point--'

    def __str__(self):
        s = f'(ParaSave) name: {self.name}, family: {self.family}\n'
        s += self.dict_2str(self.get_point())
        return s

    @property
    def logger(self):
        return self.__log