from abc import ABC
import numpy as np
from typing import Optional, Callable, List

from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.policy_gradients.base.tf_based.pg_TF_graph import pga_graph
from pypaq.neuralmess.nemodel import NEModel



class PG_TFActor(PGActor, ABC):

    def __init__(
            self,
            name: str=                      'PG_TFActor',
            nngraph: Optional[Callable]=    pga_graph,
            **kwargs):
        PGActor.__init__(
            self,
            name=       name,
            nnwrap=     NEModel,
            nngraph=    nngraph,
            **kwargs)

    def get_policy_probs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        out = self.nnw(
            feed_dict=  {self.nnw['observation_PH']: [obs_vec]},
            fetch=      ['probs'])
        return out['probs'][0] # reduce dim

    # optimized with batch call to NN
    def get_policy_probs_batch(self, observations: List[object]) -> np.ndarray:
        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.nnw(
            feed_dict=  {self.nnw['observation_PH']: obs_vecs},
            fetch=      ['probs'])
        return out['probs']

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
            fetch=      ['optimizer','probs','loss','gg_norm','gg_avt_norm','actor_ce_mean','zeroes'])
        out.pop('optimizer')
        return out