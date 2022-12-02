"""

 2022 (c) piteren

    DQNTFActor - DQNActor implemented with NEModel (TF)

"""

from abc import ABC
import numpy as np
from typing import List, Optional, Callable

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.qlearning.dqn.dqn_actor import DQN_Actor
from pypaq.R4C.qlearning.dqn.tf_based.dqn_TF_graph import dqn_graph
from pypaq.neuralmess.nemodel import NEModel


# DQN (TF NN) based QLearningActor
class DQN_TFActor(DQN_Actor, ABC):

    def __init__(self, nngraph:Optional[Callable]=dqn_graph, **kwargs):
        DQN_Actor.__init__(self, nnwrap=NEModel, nngraph=nngraph, **kwargs)

    def _get_QVs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        output = self.nnw(
            feed_dict=  {self.nnw['observations_PH']: [obs_vec]},
            fetches=    self.nnw['output'])
        return output[0] # reduce dim

    # optimized with single call to session with a batch of observations
    def get_QVs_batch(self, observations: List[object]) -> np.ndarray:
        obs_vecs = np.array([self._get_observation_vec(o) for o in observations])
        output = self.nnw(
            feed_dict=  {self.nnw['observations_PH']: obs_vecs},
            fetches=    self.nnw['output'])
        return output

    # optimized with single call to session with a batch of data
    def update_with_experience(
            self,
            observations: List[object],
            actions: List[int],
            new_qvs: List[float],
            inspect=    False) -> float:

        _, loss, gn, gn_avt = self.nnw.backward(
            feed_dict={
                self.nnw['observations_PH']: np.array([self._get_observation_vec(o) for o in observations]),
                self.nnw['enum_actions_PH']: np.array(list(enumerate(actions))),
                self.nnw['gold_QV_PH']:      np.array(new_qvs)},
            fetches=[
                self.nnw['optimizer'],
                self.nnw['loss'],
                self.nnw['gg_norm'],
                self.nnw['gg_avt_norm']])

        self._upd_step += 1

        self.nnw.log_TB(loss,    'upd/loss',     step=self._upd_step)
        self.nnw.log_TB(gn,      'upd/gn',       step=self._upd_step)
        self.nnw.log_TB(gn_avt,  'upd/gn_avt',   step=self._upd_step)

        return loss