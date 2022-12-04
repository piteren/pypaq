from abc import ABC
from typing import Optional, Callable

from pypaq.R4C.policy_gradients.base.tf_based.pg_TF_actor import PG_TFActor
from pypaq.R4C.policy_gradients.a2c.tf_based.a2c_TF_graph import a2c_graph



class A2C_TFActor(PG_TFActor, ABC):

    def __init__(
            self,
            name: str=                      'A2C_TFActor',
            nngraph: Optional[Callable]=    a2c_graph,
            **kwargs):
        PG_TFActor.__init__(
            self,
            name=       name,
            nngraph=    nngraph,
            **kwargs)

    # overrides PG_TFActor with more log_TB
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
                self.nnw['return_PH']:       dreturns},
            fetch=      ['optimizer','loss','loss_actor','loss_critic','gg_norm','gg_avt_norm','amax_prob','amin_prob','actor_ce_mean'])
        out.pop('optimizer')
        return out