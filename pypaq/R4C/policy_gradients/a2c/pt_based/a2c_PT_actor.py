from typing import Optional

from pypaq.R4C.policy_gradients.base.pt_based.pg_PT_actor import PG_PTActor
from pypaq.R4C.policy_gradients.a2c.pt_based.a2c_PT_module import A2CModel
from pypaq.torchness.motorch import Module



class A2C_PTActor(PG_PTActor):

    def __init__(
            self,
            name: str=                              'A2C_PTActor',
            module_type: Optional[type(Module)]=    A2CModel,
            **kwargs):
        PG_PTActor.__init__(
            self,
            name=           name,
            module_type=    module_type,
            **kwargs)