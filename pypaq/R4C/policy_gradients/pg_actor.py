"""

 2021 (c) piteren

    PolicyGradients NN Actor

    TODO: implement parallel training, in batches (many envys)

"""

from abc import abstractmethod, ABC
import numpy as np

from pypaq.lipytools.little_methods import stamp
from pypaq.R4C.actor import Actor
from pypaq.R4C.policy_gradients.pg_actor_graph import pga_graph
from pypaq.neuralmess.nemodel import NEModel


class PGActor(Actor, ABC):

    def __init__(
            self,
            observation,
            num_actions: int,
            mdict :dict,
            graph=      pga_graph,
            devices=    -1,
            verb=       1):

        self.verb = verb
        self.num_actions = num_actions
        observation_vec_width = self.observation_vec(observation).shape[-1]

        if self.verb>0: print(f'\n*** PGActor inits, observation_width: {observation_vec_width}')

        mdict['observation_width'] = observation_vec_width
        mdict['num_actions'] = self.num_actions
        if 'name' not in mdict: mdict['name'] = f'pgnn_{stamp()}'

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

    def get_policy_probs(self, observation) -> np.ndarray:
        ov = self.observation_vec(observation)
        probs = self.nn.session.run(
            feed_dict=  {self.nn['observation_PH']: [ov]},
            fetches=    self.nn['action_prob'])
        return probs[0]

    def get_policy_probs_batch(self, observations) -> np.ndarray:
        ovs = self.observation_vec_batch(observations)
        probs = self.nn.session.run(
            feed_dict=  {self.nn['observation_PH']: ovs},
            fetches=    self.nn['action_prob'])
        return probs

    def get_policy_action(self, observation, sampled=False) -> int:
        probs = self.get_policy_probs(observation)
        if sampled: action = np.random.choice(self.num_actions, p=probs)
        else:       action = int(np.argmax(probs))
        return action

    def get_policy_action_batch(self, observations, sampled=False) -> np.ndarray:
        probs = self.get_policy_probs_batch(observations)
        if sampled: actions = np.random.choice(self.num_actions, size=probs.shape[-1], p=probs)
        else:       actions = np.argmax(probs, axis=-1)
        return actions

    # updates self NN with batch of data
    def update_batch(
            self,
            observations,
            actions,
            dreturns     # those should be discounted accumulated returns
    ) -> float:

        ovs = self.observation_vec_batch(observations)
        _, loss, gn, gn_avt, amax_prob, amin_prob, ace = self.nn.session.run(
            fetches=    [
                self.nn['optimizer'],
                self.nn['loss'],
                self.nn['gg_norm'],
                self.nn['gg_avt_norm'],
                self.nn['amax_prob'],
                self.nn['amin_prob'],
                self.nn['actor_ce_mean']],
            feed_dict=  {
                self.nn['observation_PH']:  ovs,
                self.nn['action_PH']:       actions,
                self.nn['return_PH']:       dreturns})

        self.__upd_step += 1

        if self.verb>0:
            self.nn.log_TB(loss,        'upd/loss',             step=self.__upd_step)
            self.nn.log_TB(gn,          'upd/gn',               step=self.__upd_step)
            self.nn.log_TB(gn_avt,      'upd/gn_avt',           step=self.__upd_step)
            self.nn.log_TB(amax_prob,   'upd/amax_prob',        step=self.__upd_step)
            self.nn.log_TB(amin_prob,   'upd/amin_prob',        step=self.__upd_step)
            self.nn.log_TB(ace,         'upd/actor_ce_mean',    step=self.__upd_step)
        return loss