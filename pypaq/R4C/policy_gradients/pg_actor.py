"""

 2021 (c) piteren

    PG_Actor - PolicyGradients TrainableActor, NN based

    TODO: implement parallel training, in batches (many envys)

"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Callable, Union, List

from pypaq.R4C.actor import TrainableActor
from pypaq.R4C.envy import FiniteActionsRLEnvy
from pypaq.torchness.comoneural import NNWrap


class PGActor(TrainableActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            nnwrap: type(NNWrap),
            module_type: Optional[Union[Callable, type]]=   None,
            seed: int=                                      123,
            **kwargs):

        TrainableActor.__init__(
            self,
            envy=   envy,
            **kwargs)
        self._envy = envy  # to update type (for pycharm only)

        np.random.seed(seed)

        # some overrides and updates
        kwargs['name'] = self.name
        kwargs['name_timestamp'] = False                # name timestamp is driven by TrainableActor (with self.name)
        if 'logger' in kwargs: kwargs.pop('logger')     # NNWrap will always create own logger (since then it is not given) with optionally given level
        kwargs['num_actions'] = self._envy.num_actions()
        kwargs['observation_width'] = self._get_observation_vec(self._envy.get_observation()).shape[-1]

        self.nnw: NNWrap = nnwrap(module_type=module_type, seed=seed, **kwargs)

        self._rlog.info('*** PG_Actor *** (NN based) initialized')
        self._rlog.info(f'> NNWrap: {nnwrap.__name__}')

    # vectorization of observations batch, may be overridden with more optimal custom implementation
    def _get_observation_vec_batch(self, observations: List[object]) -> np.ndarray:
        return np.asarray([self._get_observation_vec(v) for v in observations])

    @abstractmethod
    def get_policy_probs(self, observation: object) -> np.ndarray: pass

    # baseline, may be overridden with more optimal custom implementation
    def get_policy_probs_batch(self, observations: List[object]) -> np.ndarray:
        return np.asarray([self.get_policy_probs(o) for o in observations])

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

    def _get_save_topdir(self) -> str:
        return self.nnw['save_topdir']

    def save(self):
        self.nnw.save()

    def __str__(self):
        return self.nnw.__str__()