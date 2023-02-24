from abc import ABC

from typing import Optional

from pypaq.torchness.motorch import Module
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
            observations,
            actions,
            dreturns,
            inspect=    False) -> dict:
        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.model.backward(
            observation=    obs_vecs,
            action_taken=   actions,
            qv_label=       dreturns)
        out.pop('logits')
        if 'probs' in out: out['probs'] = out['probs'].cpu().detach().numpy()
        return out