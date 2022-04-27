"""

 2020 (c) piteren

    Parameters Save

        - saves/loads DNA to/from folder (serializes object)
        - implements GX for saved

"""

import os
import shutil
from typing import Optional
import warnings

from pypaq.lipytools.little_methods import prep_folder, w_pickle, r_pickle, stamp
from pypaq.pms.base_types import POINT
from pypaq.pms.subscriptable import SubGX

SAVE_FN_PFX = 'dna' # filename (DNA) prefix
OBJ_SUFFIX = '.dct' # dna obj(pickle) filename suffix
TXT_SUFFIX = '.txt' # dna text filename suffix


class ParaSave(SubGX):

    def __init__(
            self,
            name:str,
            save_topdir: Optional[str]= None,        # ParaSave top directory, when not given folders functionality (load/save) is not accessible
            save_fn_pfx: str=           SAVE_FN_PFX, # default ParaSave filename (DNA) prefix
            gxable: bool or None=       None,        # None sets to default - True, it may override default and save by setting it to True/False
            assert_saved=               False,       # for True asserts that ParaSave has been already saved in save_topdir
            verb=                       0,
            **kwargs):

        self.name = name

        if assert_saved:
            obj_FN = ParaSave.__obj_fn(name, save_topdir, save_fn_pfx)
            assert os.path.isfile(obj_FN), f'ERR: ParaSave {self.name} does not exist!'

        self.verb = verb
        if self.verb>0: print(f'\n *** ParaSave *** name: {self.name}, initializing..')

        self.save_topdir = save_topdir
        self.save_fn_pfx = save_fn_pfx
        self.gxable = True
        self.parents = [] # list of parents names

        dna_folder = ParaSave.load_dna(
            name=           self.name,
            save_topdir=    self.save_topdir,
            save_fn_pfx=    self.save_fn_pfx)
        self.update(dna_folder) # 1. update with params from folder
        self.verb = verb        # 2. update verb in case it was saved in folder
        self.update(kwargs)     # 3. update with params given by user
        if gxable is not None: self.gxable = gxable  # if user forces it to be True/False

        SubGX.__init__(self, **self.get_point())

    # ************************************************************************************************ folder management

    @staticmethod
    def __full_dir(
            name: str,
            save_topdir: str):
        return f'{save_topdir}/{name}'

    @staticmethod
    def __obj_fn(
            name: str,
            save_topdir: str,
            save_fn_pfx: Optional[str]= SAVE_FN_PFX):
        return f'{ParaSave.__full_dir(name, save_topdir)}/{save_fn_pfx}{OBJ_SUFFIX}'

    @staticmethod
    def __txt_fn(
            name: str,
            save_topdir: str,
            save_fn_pfx: Optional[str]= SAVE_FN_PFX):
        return f'{ParaSave.__full_dir(name, save_topdir)}/{save_fn_pfx}{TXT_SUFFIX}'

    # loads ParaSave DNA from folder
    @staticmethod
    def load_dna(
            name: str,
            save_topdir: Optional[str],
            save_fn_pfx: Optional[str]= SAVE_FN_PFX) -> POINT:
        if save_topdir is None: return {}
        obj_FN = ParaSave.__obj_fn(name, save_topdir, save_fn_pfx)
        if os.path.isfile(obj_FN):
            dna: POINT = r_pickle(obj_FN)
            return dna
        return {}

    # saves ParaSave DNA to folder (with preview in txt)
    def save_dna(self):

        assert self.save_topdir, 'ERR: cannot save ParaSave, if save directory was not given, aborting!'

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

        dna: POINT = self.get_point()
        w_pickle(dna, obj_FN)
        with open(txt_FN, 'w') as file:
            s = f' *** ParaSave DNA saved: {stamp(year=True, letters=None)} ***\n'
            s += ParaSave.dict_2str(dna)
            file.write(s)

    # loads, next overrides parameters from kwargs and saves new ParaSave DNA
    @staticmethod
    def oversave(
            name: str,
            save_topdir: Optional[str],
            save_fn_pfx: Optional[str]= SAVE_FN_PFX,
            **kwargs):
        psc = ParaSave(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)
        psc.update(kwargs)
        psc.save_dna()

    # copies saved ParaSave DNA from one folder to another
    @staticmethod
    def copy_saved_dna(
            name_src: str,
            name_trg: str,
            save_topdir_src: str,
            save_topdir_trg: Optional[str] =    None,
            save_fn_pfx: Optional[str] =        SAVE_FN_PFX):

        if save_topdir_trg is None: save_topdir_trg = save_topdir_src

        ps_src = ParaSave(
            name=           name_src,
            save_topdir=    save_topdir_src,
            save_fn_pfx=    save_fn_pfx,
            assert_saved=   True)

        ps_trg = ParaSave(
            name=           name_trg,
            save_topdir=    save_topdir_trg,
            save_fn_pfx=    save_fn_pfx)

        dna_src = ps_src.get_point()
        for k in ['name','save_topdir']: dna_src.pop(k)
        ps_trg.update(dna_src)
        ps_trg.save_dna()

    # performs GX on saved ParaSave DNA (without even building child objects)
    @staticmethod
    def gx_saved_dna(
            name_parent_main: str,
            name_parent_scnd: Optional[str],                            # if not given makes GX only with main parent
            name_child: str,
            save_topdir_parent_main: str,                               # ParaSave top directory
            save_topdir_parent_scnd: Optional[str] =    None,           # ParaSave top directory of parent scnd
            save_topdir_child: Optional[str] =          None,           # ParaSave top directory of child
            save_fn_pfx: Optional[str] =                SAVE_FN_PFX,    # ParaSave dna filename prefix
    ) -> None:

        if not save_topdir_parent_scnd: save_topdir_parent_scnd = save_topdir_parent_main
        if not save_topdir_child: save_topdir_child = save_topdir_parent_main

        pm = ParaSave(
            name=           name_parent_main,
            save_topdir=    save_topdir_parent_main,
            save_fn_pfx=    save_fn_pfx,
            assert_saved=   True)

        ps = ParaSave(
            name=           name_parent_scnd,
            save_topdir=    save_topdir_parent_scnd,
            save_fn_pfx=    save_fn_pfx) if name_parent_scnd else None

        not_gxable_parents = []
        if not pm.gxable: not_gxable_parents.append(pm)
        if ps and not ps.gxable: not_gxable_parents.append(ps)
        if not_gxable_parents: warnings.warn('There are not gxable parents, cannot GX!')
        else:
            # make pm a child and save
            dna_child = SubGX.gx_dna(
                parent_main=    pm,
                parent_scnd=    ps,
                name_child=     name_child)
            pm.update(dna_child)
            pm['_save_topdir'] = save_topdir_child
            pm.parents = [name_parent_main]
            if name_parent_scnd: pm.parents.append(name_parent_scnd)
            pm.save_dna()

    # adds gxable check
    @staticmethod
    def gx_dna(
            parent_main: "ParaSave",
            parent_scnd: Optional["ParaSave"]=  None,
            name_child: Optional[str]=          None,
            **kwargs                                # here PaSpa.sample_point_GX params may be given
    ) -> POINT:
        not_gxable_parents = []
        if not parent_main.gxable: not_gxable_parents.append(parent_main)
        if parent_scnd and not parent_scnd.gxable: not_gxable_parents.append(parent_scnd)
        if not_gxable_parents: warnings.warn('There are not gxable parents, cannot GX!')
        else:
            return SubGX.gx_dna(
                parent_main=    parent_main,
                parent_scnd=    parent_scnd,
                name_child=     name_child,
                **kwargs)

    # returns difference in POINT of saved VS self
    def _get_diff_saved(self) -> POINT:
        dna_current = self.get_point()
        dna_saved = ParaSave.load_dna(
            name=           self.name,
            save_topdir=    self.save_topdir,
            save_fn_pfx=    self.save_fn_pfx)
        return {k: dna_saved[k] for k in dna_saved if dna_saved[k]!=dna_current[k]}

    # reloads and applies new values of parameters from save (allows to apply new settings after GX of saved)
    def reload(self) -> POINT:
        warnings.warn(f'NOT IMPLEMENTED reload() for {type(self)}(ParaSave)!')
        return self._get_diff_saved()

    # returns nice string of given dict (mostly for .txt preview save)
    @staticmethod
    def dict_2str(d: dict) -> str:
        if d:
            s = ''
            max_len_sk = max([len(k) for k in d.keys()])
            for k, v in sorted(d.items()): s += f'{str(k):{max_len_sk}s} : {str(v)}\n'
            return s[:-1]
        return '--empty dna--'

    def __str__(self):
        s = f'\n(ParaSave) name: {self.name}, family: {self.family}\n'
        s += self.dict_2str(self.get_point())
        return s