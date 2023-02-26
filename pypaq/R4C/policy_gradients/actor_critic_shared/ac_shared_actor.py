from abc import ABC

from typing import Optional, List, Dict, Any

from pypaq.torchness.motorch import Module
from pypaq.R4C.helpers import extract_from_batch
from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.policy_gradients.actor_critic_shared.ac_shared_actor_module import ACSharedActorModule


class ACSharedActor(PGActor, ABC):

    def __init__(
            self,
            name: str=                              'ACSharedActor',
            module_type: Optional[type(Module)]=    ACSharedActorModule,
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

        observations = extract_from_batch(batch, 'dreturns')
        obs_vecs = self._get_observation_vec_batch(observations)

        out = self.model.backward(
            observation=    obs_vecs,
            action_taken=   extract_from_batch(batch, 'actions'),
            qv_label=       extract_from_batch(batch, 'dreturns'))

        out.pop('logits')
        if 'probs' in out: out['probs'] = out['probs'].cpu().detach().numpy()

        return out