from abc import ABC, abstractmethod
from typing import Optional

from pypaq.pms.parasave import ParaSave
from pypaq.pms.base_types import POINT


class NNWrap(ParaSave, ABC):

    @staticmethod
    @abstractmethod
    def _get_def_save_topdir() -> str: pass

    @staticmethod
    @abstractmethod
    def _get_def_save_fn_pfx() -> str: pass

    @staticmethod
    def load_dna(
            name: str,
            save_topdir: Optional[str]= None,
            save_fn_pfx: Optional[str]= None) -> POINT:
        if not save_topdir: save_topdir = NNWrap._get_def_save_topdir()
        if not save_fn_pfx: save_fn_pfx = NNWrap._get_def_save_fn_pfx()
        print(save_topdir)
        print(save_fn_pfx)
        return ParaSave.load_dna(
            name=           name,
            save_topdir=    save_topdir,
            save_fn_pfx=    save_fn_pfx)