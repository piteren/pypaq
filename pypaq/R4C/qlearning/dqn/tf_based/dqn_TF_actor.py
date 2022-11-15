"""

 2022 (c) piteren

    DQNTFActor - DQNActor implemented with NEModel (TF)

"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List

from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.lipytools.little_methods import stamp
from pypaq.R4C.qlearning.qlearning_actor import QLearningActor
from pypaq.R4C.qlearning.dqn.tf_based.dqn_TF_graph import dqn_graph
from pypaq.neuralmess.nemodel import NEModel


# DQN TF (NN) based QLearningActor
class DQN_TFActor(QLearningActor, ABC):

    def __init__(
            self,
            num_actions: int,
            observation,
            mdict: dict,
            graph=          dqn_graph,
            save_topdir=    '_models',
            logger=         None,
            loglevel=       20):

        logger_given = bool(logger)
        if not logger_given:
            logger = get_pylogger(
                name=       'DQN_TFActor',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.__log = logger

        if 'name' not in mdict: mdict['name'] = f'dqnPT_{stamp()}'
        mdict['num_actions'] = num_actions
        mdict['observation_width'] = self._get_observation_vec(observation).shape[-1]
        mdict['save_topdir'] = save_topdir

        self.__log.info(f'*** DQN_TFActor {mdict["name"]} (TF based) initializes..')
        self.__log.info(f'> num_actions:       {mdict["num_actions"]}')
        self.__log.info(f'> observation_width: {mdict["observation_width"]}')
        self.__log.info(f'> save_topdir:       {mdict["save_topdir"]}')

        self.nn = NEModel(
            fwd_func=   graph,
            logger=     None if not logger_given else get_hi_child(self.__log, 'NEModel', higher_level=False),
            loglevel=   loglevel,
            **mdict)

        self._upd_step = 0

        self.__log.info(f'DQN_TFActor initialized')

    # prepares numpy vector from observation, it is a private / internal skill of Actor
    @abstractmethod
    def _get_observation_vec(self, observation: object) -> np.ndarray: pass

    def get_QVs(self, observation: object) -> np.ndarray:
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

    # INFO: not used since this Actor updates only with batches
    def upd_QV(
            self,
            observation: object,
            action: int,
            new_qv: float) -> float:
        raise Exception('not implemented')

    def save(self): self.nn.save()

    def __str__(self): return self.nn.__str__()