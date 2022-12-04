from abc import ABC

from typing import Optional, Callable

from pypaq.R4C.policy_gradients.base.tf_based.pg_TF_actor import PG_TFActor
from pypaq.R4C.policy_gradients.actor_critic_shared.tf_based.ac_shared_TF_graph import acs_graph


class ACShared_TFActor(PG_TFActor, ABC):

    def __init__(
            self,
            name: str=                      'ACShared_TFActor',
            nngraph: Optional[Callable]=    acs_graph,
            **kwargs):
        PG_TFActor.__init__(
            self,
            name=       name,
            nngraph=    nngraph,
            **kwargs)

    def update_with_experience(
            self,
            observations,
            actions,
            dreturns,
            inspect=    False) -> dict:
        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.nnw.backward(
            feed_dict=  {
                self.nnw['observation_PH']:  obs_vecs,
                self.nnw['action_PH']:       actions,
                self.nnw['qv_label_PH']:     dreturns},
            fetch=      ['optimizer','loss','loss_actor','loss_critic','gg_norm','gg_avt_norm','amax_prob','amin_prob'])
        out.pop('optimizer')
        return out