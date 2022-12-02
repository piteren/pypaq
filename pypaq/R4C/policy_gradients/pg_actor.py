"""

 2021 (c) piteren

    PG_Actor - PolicyGradients TrainableActor, NN based

    TODO: implement parallel training, in batches (many envys)

"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Callable, Union, List

from pypaq.lipytools.little_methods import stamp
from pypaq.R4C.actor import TrainableActor
from pypaq.R4C.envy import FiniteActionsRLEnvy
from pypaq.comoneural.nnwrap import NNWrap


class PGActor(TrainableActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            nnwrap: type(NNWrap),
            nngraph: Optional[Union[Callable, type]]=   None,
            name: Optional[str]=                        None,
            **kwargs):

        ta_kwargs = {k: kwargs[k] for k in ['seed','logger','loglevel'] if k in kwargs}
        TrainableActor.__init__(self, envy=envy, **ta_kwargs)
        self._envy = envy  # to update type (for pycharm only)

        kwargs['num_actions'] = self._envy.num_actions()
        kwargs['observation_width'] = self._get_observation_vec(self._envy.get_observation()).shape[-1]
        if 'logger' in kwargs: kwargs.pop('logger') #INFO: NNWrap will always create own loger with given level
        self.nnw: NNWrap = nnwrap(
            nngraph=    nngraph,
            name=       name or f'nn_{self.__class__.__name__}_{stamp()}',
            **kwargs)

        self._upd_step = 0

        self._log.info('*** PG_Actor *** (NN based) initialized')
        self._log.info(f'> NNW model name:    {self.nnw["name"]}')
        self._log.info(f'> num_actions:       {self.nnw["num_actions"]}')
        self._log.info(f'> observation_width: {self.nnw["observation_width"]}')
        self._log.info(f'> save_topdir:       {self.nnw["save_topdir"]}')

    # vectorization of observations batch, may be overridden with more optimal custom implementation
    def _get_observation_vec_batch(self, observations: List[object]) -> np.ndarray:
        return np.array([self._get_observation_vec(v) for v in observations])

    @abstractmethod
    def get_policy_probs(self, observation: object) -> np.ndarray: pass

    # baseline, may be overridden with more optimal custom implementation
    def get_policy_probs_batch(self, observations: List[object]) -> np.ndarray:
        return np.array([self.get_policy_probs(o) for o in observations])

    # gets policy action based on policy (action) probs
    def get_policy_action(self, observation: object, sampled=False) -> int:
        probs = self.get_policy_probs(observation)
        if sampled: action = np.random.choice(self._envy.num_actions(), p=probs)
        else:       action = np.argmax(probs)
        return int(action)

    def get_policy_action_batch(self, observations: List[object], sampled=False) -> np.ndarray:
        probs = self.get_policy_probs_batch(observations)
        if sampled: actions = np.random.choice(self._envy.num_actions(), size=probs.shape[-1], p=probs)
        else:       actions = np.argmax(probs, axis=-1)
        return actions

    # updates self NN with batch of data
    @abstractmethod
    def update_with_experience(
            self,
            observations,
            actions,
            dreturns,     # discounted accumulated returns
            inspect=    False) -> float: pass

    def save(self) -> None: self.nnw.save()

    def __str__(self): return self.nnw.__str__()