from abc import ABC
import numpy as np
from typing import Optional, Callable, List

from pypaq.R4C.policy_gradients.base.tf_based.pg_TF_actor import PG_TFActor
from pypaq.R4C.policy_gradients.actor_critic.tf_based.ac_TF_critic_graph import critic_graph


class AC_TFCritic(PG_TFActor, ABC):

    def __init__(self, nngraph:Optional[Callable]=critic_graph, **kwargs):
        PG_TFActor.__init__(self, nngraph=nngraph, **kwargs)

    # AC_TFCritic does not use this method
    def get_policy_probs(self, observation: object) -> np.ndarray:
        raise Exception('not implemented!')

    # AC_TFCritic does not use this method
    def get_policy_probs_batch(self, observations: List[object]) -> np.ndarray:
        raise Exception('not implemented!')

    def get_qvs(self, observation) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        out = self.nnw(
            feed_dict=  {self.nnw['observation_PH']: [obs_vec]},
            fetch=      ['qvs'])
        return out['qvs']

    def get_qvs_batch(self, observations) -> np.ndarray:
        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.nnw(
            feed_dict=  {self.nnw['observation_PH']: obs_vecs},
            fetch=      ['qvs'])
        return out['qvs']

    def update_with_experience(
            self,
            observations,
            actions_OH,
            next_action_qvs,
            next_actions_probs,
            rewards,
            inspect=    False) -> dict:

        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.nnw.backward(
            feed_dict=  {
                self.nnw['observation_PH']:       obs_vecs,
                self.nnw['action_OH_PH']:         actions_OH,
                self.nnw['next_action_qvs_PH']:   next_action_qvs,
                self.nnw['next_action_probs_PH']: next_actions_probs,
                self.nnw['reward_PH']:            rewards},
            fetch=      ['optimizer','loss','gg_norm','gg_avt_norm'])
        out.pop('optimizer')
        return out