"""

 2021 (c) piteren

    PolicyGradients NN Actor, TF based

"""

from abc import ABC
import numpy as np
from typing import Optional, Callable, List

from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.policy_gradients.base.tf_based.pg_TF_graph import pga_graph
from pypaq.neuralmess.nemodel import NEModel



class PG_TFActor(PGActor, ABC):

    def __init__(self, nngraph:Optional[Callable]=pga_graph, **kwargs):
        PGActor.__init__(self, nnwrap=NEModel, nngraph=nngraph, **kwargs)

    def get_policy_probs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        out = self.nnw(
            feed_dict=  {self.nnw['observation_PH']: [obs_vec]},
            fetch=      ['action_prob'])
        return out['action_prob'][0] # reduce dim

    # optimized with batch call to NN
    def get_policy_probs_batch(self, observations: List[object]) -> np.ndarray:
        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.nnw(
            feed_dict=  {self.nnw['observation_PH']: obs_vecs},
            fetch=      ['action_prob'])
        return out['action_prob']

    # updates self NN with batch of data
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
            fetch=      ['optimizer','loss','gg_norm','gg_avt_norm','amax_prob','amin_prob','actor_ce_mean'])
        out.pop('optimizer')
        return out