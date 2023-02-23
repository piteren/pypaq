from typing import Optional

from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.policy_gradients.a2c.a2c_actor_module import A2CModule
from pypaq.torchness.motorch import Module



class A2CActor(PGActor):

    def __init__(
            self,
            name: str=                              'A2CActor',
            module_type: Optional[type(Module)]=    A2CModule,
            **kwargs):
        PGActor.__init__(
            self,
            name=           name,
            module_type=    module_type,
            **kwargs)