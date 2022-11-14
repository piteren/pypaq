"""

 2022 (c) piteren

    DQNTFActor - DQNActor implemented with NEModel (TF)

"""

from abc import ABC
import numpy as np

from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.R4C.qlearning.dqn.dqn_actor_common import DQNActor
from pypaq.R4C.qlearning.dqn.tf_based.dqn_TF_graph import dqn_graph
from pypaq.neuralmess.nemodel import NEModel


# DQN TF (NN) based QLearningActor
class DQN_TFActor(DQNActor, ABC):

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

        mdict['save_topdir'] = save_topdir
        self.__log.info(f'DQN (TF based) Actor initializes, save_topdir: {mdict["save_topdir"]}')

        DQNActor.__init__(
            self,
            num_actions=    num_actions,
            observation=    observation,
            mdict=          mdict,
            logger=         self.__log,
            name_pfx=       'dqnTF')

        self.nn = self._init_model(
            fwd_func=   graph,
            mdict=      mdict,
            logger=     None if not logger_given else get_hi_child(self.__log, 'MOTorch', higher_level=False),
            loglevel=   loglevel)

        self._upd_step = 0

    def _init_model(self, fwd_func, mdict, logger, loglevel):
        return NEModel(
            fwd_func=   fwd_func,
            logger=     logger,
            loglevel=   loglevel,
            **mdict)

    def get_QVs(self, observation: np.ndarray) -> np.ndarray:
        output = self.nn.session.run(
            fetches=    self.nn['output'],
            feed_dict=  {self.nn['observations_PH']: [observation]})
        return output[0] # reduce dim

    # optimized with single call to session with a batch of observations
    def get_QVs_batch(self, observations: np.ndarray) -> np.ndarray:
        output = self.nn.session.run(
            fetches=    self.nn['output'],
            feed_dict=  {self.nn['observations_PH']: observations})
        return output

    # optimized with single call to session with a batch of data
    def update_with_experience(
            self,
            observations: np.ndarray,
            actions: np.ndarray,
            new_qvs: np.ndarray) -> float:

        _, loss, gn, gn_avt = self.nn.session.run(
            fetches=[
                self.nn['optimizer'],
                self.nn['loss'],
                self.nn['gg_norm'],
                self.nn['gg_avt_norm']],
            feed_dict={
                self.nn['observations_PH']: observations,
                self.nn['enum_actions_PH']: np.array(list(enumerate(actions))),
                self.nn['gold_QV_PH']:      new_qvs})

        self._upd_step += 1

        self.nn.log_TB(loss,    'upd/loss',     step=self._upd_step)
        self.nn.log_TB(gn,      'upd/gn',       step=self._upd_step)
        self.nn.log_TB(gn_avt,  'upd/gn_avt',   step=self._upd_step)

        return loss