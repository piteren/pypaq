"""

 2022 (c) piteren

    DQNN - QLearningActor based on NN (NEModel)

    Similar to QLearningActor, DQNN has its own QV update override implemented with NN Optimizer that is responsible by
    updating NN weights with call of optimizer policies like learning ratio and other.
    DQNN updates not QV but NN weights according to loss calculated with given gold QV.

"""

from abc import ABC
import numpy as np

from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.R4C.qlearning.dqn.dqn_actor_common import DQNActor
from pypaq.R4C.qlearning.dqn.tf_based.dqn_TF_graph import dqn_graph
from pypaq.neuralmess.nemodel import NEModel


# QLearningActor with NeuralNetwork (DQN)
class DQNTFActor(DQNActor, ABC):

    def __init__(
            self,
            num_actions: int,
            observation,
            mdict: dict,
            graph=      dqn_graph,
            logger=     None,
            loglevel=   20):

        if not logger:
            logger = get_pylogger(
                name=       'DQNTFActor',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.__log = logger

        DQNActor.__init__(
            self,
            num_actions=    num_actions,
            observation=    observation,
            mdict=          mdict,
            logger=         get_hi_child(self.__log, 'DQNActor', higher_level=False))

        self.nn = self._init_model(
            fwd_func=   graph,
            mdict=      mdict)

        self._upd_step = 0

    def _init_model(self, fwd_func, mdict):
        return NEModel(
            fwd_func=       fwd_func,
            save_topdir=    '_models',
            verb=           0,
            **mdict)

    def get_QVs(self, observation) -> np.ndarray:
        ov = self.get_observation_vec(observation)
        output = self.nn.session.run(
            fetches=    self.nn['output'],
            feed_dict=  {self.nn['observations_PH']: [ov]})
        return output[0] # reduce dim

    # optimized with single call to session with a batch of observations
    def get_QVs_batch(self, observations) -> np.ndarray:
        ovs = [self.get_observation_vec(observation) for observation in observations]
        output = self.nn.session.run(
            fetches=    self.nn['output'],
            feed_dict=  {self.nn['observations_PH']: np.array(ovs)})
        return output

    # optimized with single call to session with a batch of data
    def update_with_experience(
            self,
            observations: np.ndarray,
            actions: np.ndarray,
            new_qvs: np.ndarray) -> float:

        ovs = [self.get_observation_vec(observation) for observation in observations]
        _, loss, gn, gn_avt = self.nn.session.run(
            fetches=[
                self.nn['optimizer'],
                self.nn['loss'],
                self.nn['gg_norm'],
                self.nn['gg_avt_norm']],
            feed_dict={
                self.nn['observations_PH']:  np.array(ovs),
                self.nn['enum_actions_PH']: np.array(list(enumerate(actions))),
                self.nn['gold_QV_PH']:      np.array(new_qvs)})

        self._upd_step += 1

        self.nn.log_TB(loss,    'upd/loss',     step=self._upd_step)
        self.nn.log_TB(gn,      'upd/gn',       step=self._upd_step)
        self.nn.log_TB(gn_avt,  'upd/gn_avt',   step=self._upd_step)

        return loss