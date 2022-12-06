from typing import Optional

from pypaq.R4C.policy_gradients.base.pt_based.pg_PT_actor import PG_PTActor
from pypaq.R4C.policy_gradients.a2c.pt_based.a2c_PT_module import A2CModel
from pypaq.torchness.motorch import Module



class A2C_PTActor(PG_PTActor):

    def __init__(
            self,
            name: str=                          'A2C_PTActor',
            nngraph: Optional[type(Module)]=    A2CModel,
            **kwargs):
        PG_PTActor.__init__(
            self,
            name=       name,
            nngraph=    nngraph,
            **kwargs)

    # removes 'value'
    def update_with_experience(
            self,
            observations,
            actions,
            dreturns,
            inspect=    False) -> dict:
        out = PG_PTActor.update_with_experience(
            self,
            observations=   observations,
            actions=        actions,
            dreturns=       dreturns,
            inspect=        inspect)
        if 'value' in out: out.pop('value')
        return out