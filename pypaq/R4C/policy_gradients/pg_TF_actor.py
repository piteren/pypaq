"""

 2021 (c) piteren

    PolicyGradients NN Actor, TF based

    TODO: implement parallel training, in batches (many envys)

"""

from abc import ABC
import numpy as np

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.little_methods import stamp
from pypaq.R4C.actor import TrainableActor
from pypaq.R4C.envy import FiniteActionsRLEnvy
from pypaq.R4C.policy_gradients.pg_TF_graph import pga_graph
from pypaq.neuralmess.nemodel import NEModel


class PG_TFActor(TrainableActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            mdict :dict,
            graph=          pga_graph,
            save_topdir=    '_models',
            logger=         None,
            loglevel=       20):

        logger_given = bool(logger)
        if not logger_given:
            logger = get_pylogger(
                name=       self.__class__.__name__,
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.__log = logger

        if 'name' not in mdict: mdict['name'] = f'pgTF_{stamp()}'
        self._envy = envy
        mdict['num_actions'] = self._envy.num_actions()
        mdict['observation_width'] = self._get_observation_vec(self._envy.get_observation()).shape[-1]
        mdict['save_topdir'] = save_topdir

        self.__log.info('*** PG_TFActor (TF based) initializes..')
        self.__log.info(f'> Envy:              {self._envy.__class__.__name__}')
        self.__log.info(f'> NN model name:     {mdict["name"]}')
        self.__log.info(f'> num_actions:       {mdict["num_actions"]}')
        self.__log.info(f'> observation_width: {mdict["observation_width"]}')
        self.__log.info(f'> save_topdir:       {mdict["save_topdir"]}')

        self.nn = NEModel(
            fwd_func=   graph,
            logger=     None if not logger_given else self.__log,
            loglevel=   loglevel,
            **mdict)

        self.__upd_step = 0

    # vectorization of observations batch, may be overridden with more optimal custom implementation
    def observation_vec_batch(self, observations) -> np.ndarray:
        return np.array([self._get_observation_vec(v) for v in observations])

    def get_policy_probs(self, observation) -> np.ndarray:
        ov = self._get_observation_vec(observation)
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
        if sampled: action = np.random.choice(self._envy.num_actions(), p=probs)
        else:       action = np.argmax(probs)
        return int(action)

    def get_policy_action_batch(self, observations, sampled=False) -> np.ndarray:
        probs = self.get_policy_probs_batch(observations)
        if sampled: actions = np.random.choice(self._envy.num_actions(), size=probs.shape[-1], p=probs)
        else:       actions = np.argmax(probs, axis=-1)
        return actions

    # updates self NN with batch of data
    def update_with_experience(
            self,
            observations,
            actions,
            dreturns,     # discounted accumulated returns
            inspect=    False) -> float:

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

        self.nn.log_TB(loss,        'upd/loss',             step=self.__upd_step)
        self.nn.log_TB(gn,          'upd/gn',               step=self.__upd_step)
        self.nn.log_TB(gn_avt,      'upd/gn_avt',           step=self.__upd_step)
        self.nn.log_TB(amax_prob,   'upd/amax_prob',        step=self.__upd_step)
        self.nn.log_TB(amin_prob,   'upd/amin_prob',        step=self.__upd_step)
        self.nn.log_TB(ace,         'upd/actor_ce_mean',    step=self.__upd_step)

        return loss

    def __str__(self): return self.nn.__str__()