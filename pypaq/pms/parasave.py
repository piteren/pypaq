import os
import shutil
from typing import Optional, List

from pypaq.exception import PyPaqException
from pypaq.lipytools.printout import stamp
from pypaq.lipytools.files import r_pickle, w_pickle, prep_folder
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.pms.base import POINT
from pypaq.pms.para import ParaGX


class ParaSaveException(PyPaqException):
    pass


class ParaSave(ParaGX):
    """ Parameters Save
    saves/loads POINT (self params) from save_topdir
    implements GX for saved

    folders functionality (load, save ..) is disabled for save_topdir / SAVE_TOPDIR == None

    ParaSave defaults are stored in PARASAVE_DEFAULTS dict and cannot be placed in __init__ defaults.
    This is a consequence of the params resolution mechanism in ParaSave,
    where parameters may come from three sources, and each subsequent source overrides the previous ones:
        1. class __init__ defaults
        2. saved in the folder
        3. given with kwargs in __init__
    If all ParaSave parameters were set with __init__ defaults,
    it would not be possible to distinguish between sources 1 and 3.

    @DynamicAttrs <-- disables warning for unresolved attributes references """

    PARASAVE_DEFAULTS = {
        'gxable':   True,
        'parents':  [],     # list (nested) of parents names
    }

    SAVE_TOPDIR = None      # save top directory
    SAVE_FN_PFX = 'point'   # POINT file prefix
    OBJ_SUFFIX = '.dct'     # POINT file suffix of obj(pickle)
    TXT_SUFFIX = '.txt'     # POINT file suffix of text

    def __init__(
            self,
            name: str,
            save_topdir: Optional[str]= None,   # ParaSave top directory
            save_fn_pfx: Optional[str]= None,   # ParaSave POINT file prefix
            assert_saved=               False,  # for True asserts that ParaSave has been already saved in save_topdir
            lock_managed_params=        False,  # locks _managed_params to only those known while init
            logger=                     None,
            loglevel=                   20,
            **kwargs,
    ):
        """ name, save_topdir, save_fn_pfx -> if given -> always override saved values """

        # _managed_params allows to lock ParaSave managed params only to those resolved here (while __init__)
        self._managed_params: Optional[List[str]] = None

        self.name = name
        self.save_topdir = save_topdir or self.SAVE_TOPDIR
        self.save_fn_pfx = save_fn_pfx or self.SAVE_FN_PFX

        if not logger:
            logger = get_pylogger(
                name=       self.name,
                add_stamp=  False,
                folder=     ParaSave._full_dir(name=self.name, save_topdir=self.save_topdir) if self.save_topdir else None,
                level=      loglevel)
        self.__log = logger

        self.__log.info(f'*** ParaSave : {self.name} *** initializes..')
        self.__log.debug(f'> save_topdir: {self.save_topdir}')
        self.__log.debug(f'> save_fn_pfx: {self.save_fn_pfx}')

        if assert_saved and not self.is_saved(
                name=           self.name,
                save_topdir=    self.save_topdir,
                save_fn_pfx=    self.save_fn_pfx):
            ex_msg = f'ParaSave {self.name} does not exist, but should!'
            self.__log.error(ex_msg)
            raise ParaSaveException(ex_msg)

        point_saved = self.load_point(
            name=           self.name,
            save_topdir=    self.save_topdir,
            save_fn_pfx=    self.save_fn_pfx)

        # update in proper order
        _self_point = {}
        _self_point.update(self.PARASAVE_DEFAULTS)
        _self_point.update(point_saved)
        _self_point.update(self.get_point()) # params added up to now to self
        _self_point.update(kwargs)

        self.__log.debug(f'ParaSave POINT sources:')
        self.__log.debug(f'> PARASAVE_DEFAULTS:     {self.PARASAVE_DEFAULTS}')
        self.__log.debug(f'> POINT saved:           {point_saved}')
        self.__log.debug(f'> given kwargs:          {kwargs}')
        self.__log.debug(f'ParaSave complete POINT: {_self_point}')

        super().__init__(**_self_point)

        if lock_managed_params:
            self._managed_params = self.get_managed_params()
            self.__log.debug(f'locked managed params: {self._managed_params}')

    def get_managed_params(self) -> List[str]:
        if self._managed_params is not None:
            return self._managed_params
        return ParaGX.get_managed_params(self)

    def __setitem__(self, key, value):
        if self._managed_params is not None:
            msg = f'ParaSave (managed_params locked) asked to self set with: >{key}:{value}< self will be updated, but managed_params not!'
            self.__log.warning(msg)
        setattr(self, key, value)

    # ************************************************************************************************ folder management

    @classmethod
    def _full_dir(
            cls,
            name: str,
            save_topdir: Optional[str]= None,
    ):
        if not save_topdir: save_topdir = cls.SAVE_TOPDIR
        return f'{save_topdir}/{name}'

    @classmethod
    def _obj_fn(
            cls,
            name: str,
            save_topdir: Optional[str]= None,
            save_fn_pfx: Optional[str]= None,
    ):
        if not save_topdir: save_topdir = cls.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX
        return f'{cls._full_dir(name=name, save_topdir=save_topdir)}/{save_fn_pfx}{cls.OBJ_SUFFIX}'

    @classmethod
    def _txt_fn(
            cls,
            name: str,
            save_topdir: Optional[str]= None,
            save_fn_pfx: Optional[str]= None,
    ):
        if not save_topdir: save_topdir = cls.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX
        return f'{cls._full_dir(name=name, save_topdir=save_topdir)}/{save_fn_pfx}{cls.TXT_SUFFIX}'

    @classmethod
    def is_saved(
            cls,
            name: str,
            save_topdir: Optional[str]= None,
            save_fn_pfx: Optional[str]= None,
    ) -> bool:
        """ checks if given ParaSave name is already saved (..has been created before) """
        if not save_topdir: save_topdir = cls.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX
        return os.path.isfile(cls._obj_fn(name=name, save_topdir=save_topdir, save_fn_pfx=save_fn_pfx))

    @classmethod
    def load_point(
            cls,
            name: str,
            save_topdir: Optional[str]= None,
            save_fn_pfx: Optional[str]= None,
    ) -> POINT:
        """ loads POINT from folder """
        obj_FN = cls._obj_fn(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)
        if os.path.isfile(obj_FN):
            point: POINT = r_pickle(obj_FN)
            return point
        return {}

    def save_point(self):
        """ saves self POINT to folder (with preview in txt) """

        if not self.save_topdir:
            msg = f'cannot save {self.__class__.__name__}, if save directory was not given, aborting'
            self.__log.error(msg)
            raise ParaSaveException(msg)

        obj_FN = self._obj_fn(
            name=           self.name,
            save_topdir=    self.save_topdir,
            save_fn_pfx=    self.save_fn_pfx)
        txt_FN = self._txt_fn(
            name=           self.name,
            save_topdir=    self.save_topdir,
            save_fn_pfx=    self.save_fn_pfx)

        prep_folder(self._full_dir(
            name=           self.name,
            save_topdir=    self.save_topdir))

        if os.path.isfile(obj_FN): shutil.copy(obj_FN, f'{obj_FN}_OLD')
        if os.path.isfile(txt_FN): shutil.copy(txt_FN, f'{txt_FN}_OLD')

        point: POINT = self.get_point()
        w_pickle(point, obj_FN)
        with open(txt_FN, 'w') as file:
            s = f' *** ParaSave POINT saved: {stamp(year=True, letters=None)} ***\n'
            s += self.dict_2str(point)
            file.write(s)

        self.__log.debug(f'{self.__class__.__name__} {self.name} saved to {self.save_topdir}')

    @classmethod
    def oversave_point(
            cls,
            name: str,
            save_topdir: Optional[str]= None,
            save_fn_pfx: Optional[str]= None,
            **kwargs,
    ):
        """ loads, next overrides parameters from given kwargs and saves POINT """
        psc = cls(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx,
            loglevel=       30)
        psc.update(kwargs)
        psc.save_point()

    @classmethod
    def copy_saved_point(
            cls,
            name_src: str,
            name_trg: str,
            save_topdir_src: Optional[str]= None,
            save_topdir_trg: Optional[str]= None,
            save_fn_pfx: Optional[str]=     None,
            logger=                         None,
            loglevel=                       20,
    ) -> None:
        """ copies saved ParaSave POINT from one folder to another """

        if not save_topdir_src: save_topdir_src = cls.SAVE_TOPDIR
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX

        ps_src = cls(
            name=           name_src,
            save_topdir=    save_topdir_src,
            save_fn_pfx=    save_fn_pfx,
            assert_saved=   True,
            logger=         logger,
            loglevel=       loglevel)

        ps_src.name = name_trg
        if save_topdir_trg:
            ps_src.save_topdir = save_topdir_trg

        ps_src.save_point()

    @staticmethod
    def _gxable_check(
            parentA: "ParaSave",
            parentB: Optional["ParaSave"],
    ) -> bool:

        not_gxable_parents = []
        if not parentA['gxable']:
            not_gxable_parents.append(parentA)
        if parentB and not parentB['gxable']:
            not_gxable_parents.append(parentB)

        if not_gxable_parents:
            return False
        return True

    @classmethod
    def gx_saved_point(
            cls,
            name_parentA: str,
            name_parentB: Optional[str],                # if not given makes GX only with parent A
            name_child: str,
            save_topdir_parentA: Optional[str]= None,   # ParaSave top directory of parent A
            save_topdir_parentB: Optional[str]= None,   # ParaSave top directory of parent B
            save_topdir_child: Optional[str]=   None,   # ParaSave top directory of child
            save_fn_pfx: Optional[str]=         None,   # ParaSave POINT file prefix
            logger=                             None,
            loglevel=                           20,
    ) -> None:
        """ performs GX on saved ParaSave POINT """

        if not save_topdir_parentA: save_topdir_parentA = cls.SAVE_TOPDIR
        if not save_topdir_parentB: save_topdir_parentB = save_topdir_parentA
        if not save_topdir_child: save_topdir_child = save_topdir_parentA
        if not save_fn_pfx: save_fn_pfx = cls.SAVE_FN_PFX

        parentA = cls(
            name=           name_parentA,
            save_topdir=    save_topdir_parentA,
            save_fn_pfx=    save_fn_pfx,
            assert_saved=   True,
            logger=         logger,
            loglevel=       loglevel)

        parentB = cls(
            name=           name_parentB,
            save_topdir=    save_topdir_parentB,
            save_fn_pfx=    save_fn_pfx,
            assert_saved=   True,
            logger=         logger,
            loglevel=       loglevel) if name_parentB else None

        if not cls._gxable_check(parentA, parentB):
            raise ParaSaveException('not gxable parents, cannot GX!')

        # make child and save
        point_child = cls.gx_point(
            parentA=    parentA,
            parentB=    parentB,
            name_child= name_child)
        child = parentA
        child.update(point_child)
        child.save_topdir = save_topdir_child
        child.save_point()

    @staticmethod
    def gx_point(
            parentA: "ParaSave",
            parentB: Optional["ParaSave"]=  None,
            name_child: Optional[str]=      None,
            prob_mix=                       0.5,
            prob_noise=                     0.3,
            noise_scale=                    0.1,
            prob_axis=                      0.1,
            prob_diff_axis=                 0.3,
    ) -> POINT:
        """ adds gxable check """

        if not ParaSave._gxable_check(parentA, parentB):
            raise ParaSaveException('not gxable parents, cannot GX')

        child_point = ParaGX.gx_point(
            parentA=        parentA,
            parentB=        parentB,
            name_child=     name_child,
            prob_mix=       prob_mix,
            prob_noise=     prob_noise,
            noise_scale=    noise_scale,
            prob_axis=      prob_axis,
            prob_diff_axis= prob_diff_axis)

        child_point['parents'] = [parentA.parents if parentA.parents else parentA.name]
        if parentB:
            child_point['parents'].append(parentB.parents if parentB.parents else parentB.name)

        return child_point

    @property
    def logger(self):
        return self.__log

    @staticmethod
    def dict_2str(d:dict) -> str:
        """ returns nice string of given dict (mostly for .txt preview save) """
        if d:
            s = ''
            max_len_sk = max([len(k) for k in d.keys()])
            for k, v in sorted(d.items()): s += f'{str(k):{max_len_sk}s} : {str(v)}\n'
            return s[:-1]
        return '--empty point--'

    def __str__(self):
        s = f'{self.__class__.__name__} (ParaSave)\n'
        s += self.dict_2str(self.get_point())
        return s
