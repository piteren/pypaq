"""

 2022 (c) piteren

    DQNTFActor - DQNActor implemented with NEModel (TF)

"""

from abc import ABC
import numpy as np
from typing import List

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.qlearning.dqn.dqn_actor import DQN_Actor
from pypaq.R4C.qlearning.dqn.tf_based.dqn_TF_graph import dqn_graph
from pypaq.neuralmess.nemodel import NEModel


# DQN (TF NN) based QLearningActor
class DQN_TFActor(DQN_Actor, ABC):

    def __init__(
            self,
            mdict: dict,
            graph=          dqn_graph,
            save_topdir=    '_models',
            logger=         None,
            loglevel=       20,
            **kwargs):

        self._logger_given = bool(logger)
        self._loglevel = loglevel
        if not self._logger_given:
            logger = get_pylogger(
                name=       self.__class__.__name__,
                add_stamp=  True,
                folder=     save_topdir,
                level=      self._loglevel)
        self.__log = logger
        self.__log.info('*** DQN_TFActor (TF based) initializes..')
        self.__log.info(f'> graph: {graph.__name__}')

        self._graph = graph
        self._mdict = mdict

        DQN_Actor.__init__(
            self,
            mdict=          self._mdict,
            save_topdir=    save_topdir,
            logger=         self.__log,
            **kwargs)

        self.__log.info(f'DQN_TFActor initialized')

    def _get_model(self):
        return NEModel(
            fwd_func=   self._graph,
            logger=     None if not self._logger_given else self.__log,
            loglevel=   self._loglevel,
            **self._mdict)

    def _get_QVs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        output = self.nn.session.run(
            fetches=    self.nn['output'],
            feed_dict=  {self.nn['observations_PH']: [obs_vec]})
        return output[0] # reduce dim

    # optimized with single call to session with a batch of observations
    def get_QVs_batch(self, observations: List[object]) -> np.ndarray:
        obs_vecs = np.array([self._get_observation_vec(o) for o in observations])
        output = self.nn.session.run(
            fetches=    self.nn['output'],
            feed_dict=  {self.nn['observations_PH']: obs_vecs})
        return output

    # optimized with single call to session with a batch of data
    def update_with_experience(
            self,
            observations: List[object],
            actions: List[int],
            new_qvs: List[float],
            inspect=    False) -> float:

        _, loss, gn, gn_avt = self.nn.session.run(
            fetches=[
                self.nn['optimizer'],
                self.nn['loss'],
                self.nn['gg_norm'],
                self.nn['gg_avt_norm']],
            feed_dict={
                self.nn['observations_PH']: np.array([self._get_observation_vec(o) for o in observations]),
                self.nn['enum_actions_PH']: np.array(list(enumerate(actions))),
                self.nn['gold_QV_PH']:      np.array(new_qvs)})

        self._upd_step += 1

        self.nn.log_TB(loss,    'upd/loss',     step=self._upd_step)
        self.nn.log_TB(gn,      'upd/gn',       step=self._upd_step)
        self.nn.log_TB(gn_avt,  'upd/gn_avt',   step=self._upd_step)

        return loss

    def save(self): self.nn.save()

    def __str__(self): return self.nn.__str__()