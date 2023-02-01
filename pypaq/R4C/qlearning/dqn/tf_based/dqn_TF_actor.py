"""

 2022 (c) piteren

    DQNTFActor - DQNActor implemented with NEModel (TF)

"""

import numpy as np
from typing import List, Optional, Callable

from pypaq.R4C.qlearning.dqn.dqn_actor import DQN_Actor
from pypaq.R4C.qlearning.dqn.tf_based.dqn_TF_graph import dqn_graph
from pypaq.neuralmess.nemodel import NEModel


# DQN (TF NN) based QLearningActor
class DQN_TFActor(DQN_Actor):

    def __init__(
            self,
            name: str=                      'DQN_TFActor',
            nngraph: Optional[Callable]=    dqn_graph,
            **kwargs):
        DQN_Actor.__init__(
            self,
            name=       name,
            nnwrap=     NEModel,
            nngraph=    nngraph,
            **kwargs)

    def _get_QVs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        out = self.nnw(
            feed_dict=  {self.nnw['observations_PH']: [obs_vec]},
            fetch=      ['output'])
        return out['output'][0] # reduce dim

    # optimized with single call to session with a batch of observations
    def get_QVs_batch(self, observations: List[object]) -> np.ndarray:
        obs_vecs = np.asarray([self._get_observation_vec(o) for o in observations])
        out = self.nnw(
            feed_dict=  {self.nnw['observations_PH']: obs_vecs},
            fetch=      ['output'])
        return out['output']

    # optimized with single call to session with a batch of data
    def update_with_experience(
            self,
            observations: List[object],
            actions: List[int],
            new_qvs: List[float],
            inspect=    False) -> dict:
        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.nnw.backward(
            feed_dict=  {
                self.nnw['observations_PH']: obs_vecs,
                self.nnw['enum_actions_PH']: np.asarray(list(enumerate(actions))),
                self.nnw['gold_QV_PH']:      np.asarray(new_qvs)},
            fetch=      ['optimizer','loss','gg_norm','gg_avt_norm'])
        out.pop('optimizer')
        return out