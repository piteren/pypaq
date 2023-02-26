from typing import Optional, List, Dict, Any

from pypaq.lipytools.plots import two_dim_multi
from pypaq.R4C.helpers import extract_from_batch
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

    def update_with_experience(
            self,
            batch: List[Dict[str, Any]],
            inspect: bool
    ) -> Dict[str, Any]:

        actor_metrics = super().update_with_experience(batch,inspect)

        if inspect:
            two_dim_multi(
                ys=     [
                    extract_from_batch(batch, 'dreturn'),
                    actor_metrics.pop('value'),
                    actor_metrics.pop('advantage'),
                ],
                names=  [
                    'dreturns',
                    'value',
                    'advantage',
                ],
                legend_loc= 'lower left')

        return actor_metrics