"""

 2021 (c) piteren

    PG_Actor - PolicyGradients TrainableActor, NN based

    TODO: implement parallel training, in batches (many envys)

"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Callable, Union, List

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.little_methods import stamp
from pypaq.R4C.actor import TrainableActor
from pypaq.R4C.envy import FiniteActionsRLEnvy
from pypaq.comoneural.nnwrap import NNWrap


class PG_Actor(TrainableActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            nnwrap: type(NNWrap),
            seed: int,
            logger,
            loglevel,
            nngraph: Optional[Union[Callable, type]]=   None,
            name: Optional[str]=                        None,
            **kwargs):

        logger_given = bool(logger)
        if not logger_given:
            logger = get_pylogger(
                name=       'PG_Actor',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.__log = logger

        TrainableActor.__init__(
            self,
            envy=   envy,
            seed=   seed,
            logger= self.__log)
        self._envy = envy  # to update type (for pycharm only)

        kwargs['num_actions'] = self._envy.num_actions()
        kwargs['observation_width'] = self._get_observation_vec(self._envy.get_observation()).shape[-1]

        self.nnw: NNWrap = nnwrap(
            nngraph=    nngraph,
            name=       name or f'nn_{self.__class__.__name__}_{stamp()}',
            logger=     self.__log if logger_given else None,
            loglevel=   loglevel,
            **kwargs)

        self._upd_step = 0

        self.__log.info('*** PG_Actor *** (NN based) initialized')
        self.__log.info(f'> NNW model name:    {self.nnw["name"]}')
        self.__log.info(f'> num_actions:       {self.nnw["num_actions"]}')
        self.__log.info(f'> observation_width: {self.nnw["observation_width"]}')
        self.__log.info(f'> save_topdir:       {self.nnw["save_topdir"]}')

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