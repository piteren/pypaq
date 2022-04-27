from abc import abstractmethod, ABC
import numpy as np


from pypaq.lipytools.little_methods import stamp
from pypaq.R4C.policy_gradients.actor_critic.ac_critic_graph import critic_graph
from pypaq.neuralmess.nemodel import NEModel


class ACCritic(ABC):

    def __init__(
            self,
            observation,
            num_actions: int,
            mdict :dict,
            graph=      critic_graph,
            devices=    0,
            verb=       1):

        self.verb = verb

        self.__num_actions = num_actions
        observation_width = self.observation_vec(observation).shape[-1]

        if self.verb>0: print(f'\n*** ACCritic inits, observation_width: {observation_width}')

        mdict['observation_width'] = observation_width
        mdict['num_actions'] = self.__num_actions
        if 'name' not in mdict: mdict['name'] = f'accnn_{stamp()}'

        self.nn = NEModel(
            fwd_func=       graph,
            devices=        devices,
            save_topdir=    '_models',
            verb=           self.verb-1,
            **mdict)

        self.__upd_step = 0

    # returns vector of feats from given observation (encodes observation into input vector)
    @abstractmethod
    def observation_vec(self, observation) -> np.ndarray: pass

    # vectorization of observations batch, may be overridden with more optimal custom implementation
    def observation_vec_batch(self, observations) -> np.ndarray:
        return np.array([self.observation_vec(v) for v in observations])

    def get_qvs(self, observation) -> np.ndarray:
        ov = self.observation_vec(observation)
        qvs = self.nn.session.run(
            feed_dict=  {self.nn['observation_PH']: [ov]},
            fetches=    self.nn['qvs'])
        return qvs

    def get_qvs_batch(self, observations) -> np.ndarray:
        ovs = self.observation_vec_batch(observations)
        qvss = self.nn.session.run(
            feed_dict=  {self.nn['observation_PH']: ovs},
            fetches=    self.nn['qvs'])
        return qvss

    def update_batch(
            self,
            observations,
            actions_OH,
            next_action_qvs,
            next_actions_probs,
            rewards) -> float:

        ovs = self.observation_vec_batch(observations)
        _, loss, gn, gn_avt = self.nn.session.run(
            fetches=    [
                self.nn['optimizer'],
                self.nn['loss'],
                self.nn['gg_norm'],
                self.nn['gg_avt_norm']],
            feed_dict=  {
                self.nn['observation_PH']:       ovs,
                self.nn['action_OH_PH']:         actions_OH,
                self.nn['next_action_qvs_PH']:   next_action_qvs,
                self.nn['next_action_probs_PH']: next_actions_probs,
                self.nn['reward_PH']:            rewards})

        self.__upd_step += 1

        if self.verb>0:
            self.nn.log_TB(loss,    'critic/loss',     step=self.__upd_step)
            self.nn.log_TB(gn,      'critic/gn',       step=self.__upd_step)
            self.nn.log_TB(gn_avt,  'critic/gn_avt',   step=self.__upd_step)

        return loss