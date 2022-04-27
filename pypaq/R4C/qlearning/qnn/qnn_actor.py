"""

 2022 (c) piteren

    DQNN - QLearningActor based on NN (NEModel)

    Similar to QLearningActor, DQNN has its own QV update override implemented with NN Optimizer that is responsible by
    updating NN weights with call of optimizer policies like learning ratio and other.
    DQNN updates not QV but NN weights according to loss calculated with given gold QV.

"""

from abc import abstractmethod, ABC
import numpy as np

from pypaq.lipytools.little_methods import stamp
from pypaq.R4C.qlearning.qlearning_actor import QLearningActor
from pypaq.R4C.qlearning.qnn.qnn_graph import qnn_graph
from pypaq.neuralmess.nemodel import NEModel


# QLearningActor with NeuralNetwork (DQNN)
class DQNActor(QLearningActor, ABC):

    def __init__(
            self,
            num_actions: int,
            observation,
            mdict: dict,
            graph=      qnn_graph,
            verb=       1):

        self.verb = verb
        mdict['num_actions'] = num_actions
        observation_width = self.observation_vec(observation).shape[-1]
        mdict['observation_width'] = observation_width
        if 'name' not in mdict: mdict['name'] = f'qnn_{stamp()}'

        if self.verb>0: print(f'\n*** DQNActor inits, num_actions: {num_actions}, observation_width: {observation_width}')

        self.nn = NEModel(
            fwd_func=       graph,
            save_topdir=    '_models',
            verb=           self.verb-1,
            **mdict)

        self.__upd_step = 0

    # returns vector of feats from given observation (encodes observation into input vector)
    @abstractmethod
    def observation_vec(self, observation) -> np.ndarray: pass

    def get_QVs(self, observation) -> np.ndarray:
        ov = self.observation_vec(observation)
        output = self.nn.session.run(
            fetches=    self.nn['output'],
            feed_dict=  {self.nn['observations_PH']: [ov]})
        return output[0] # reduce dim

    def get_QVs_batch(self, observations):
        ovs = [self.observation_vec(observation) for observation in observations]
        output = self.nn.session.run(
            fetches=    self.nn['output'],
            feed_dict=  {self.nn['observations_PH']: np.array(ovs)})
        return output

    # INFO: not used since DQNActor updates only with batches
    def upd_QV(self, observation, action, new_qv):
        raise NotImplementedError
        pass

    def update_batch(
            self,
            observations,
            actions,
            new_qvs):

        ovs = [self.observation_vec(observation) for observation in observations]
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

        self.__upd_step += 1

        if self.verb>0:
            self.nn.log_TB(loss,    'upd/loss',     step=self.__upd_step)
            self.nn.log_TB(gn,      'upd/gn',       step=self.__upd_step)
            self.nn.log_TB(gn_avt,  'upd/gn_avt',   step=self.__upd_step)