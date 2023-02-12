"""

 2022 (c) piteren

    RL Actor

"""

from abc import abstractmethod, ABC
import numpy as np
from typing import Optional

from pypaq.lipytools.printout import stamp
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.envy import RLEnvy
from pypaq.R4C.helpers import RLException


# just Actor
class Actor(ABC):

    # returns Actor action based on observation according to Actor policy
    @abstractmethod
    def get_policy_action(self, observation: object) -> object: pass

# cooperates with RLTrainer, prepares obs_vec, updates, saves
class TrainableActor(Actor, ABC):

    def __init__(
            self,
            envy: RLEnvy,
            name: str=              'TrainableActor',
            name_timestamp: bool=   True,
            logger: Optional=       None,
            loglevel: int=          20,
            **kwargs):

        if name_timestamp: name += f'_{stamp()}'
        self.name = name
        self._envy = envy

        self._rlog = logger or get_pylogger(level=loglevel)
        self._rlog.info(f'*** TrainableActor *** initialized')
        self._rlog.info(f'> name:              {self.name}')
        self._rlog.info(f'> Envy:              {self._envy.__class__.__name__}')
        self._rlog.info(f'> observation width: {self._get_observation_vec(self._envy.get_observation()).shape[-1]}')
        self._rlog.info(f'> not used kwargs:   {kwargs}')

    # prepares numpy vector from observation, first tries to get from RLEnvy
    def _get_observation_vec(self, observation: object) -> np.ndarray:
        try: return self._envy.prep_observation_vec(observation)
        except RLException: raise RLException ('TrainableActor not implemented _get_observation_vec()')

    # add sampling (from probability?) option which may be helpful for training
    @abstractmethod
    def get_policy_action(self, observation:object, sampled=False) -> object: pass

    # updates self with (batch of) experience data (given with kwargs), returns dict with Actor "metrics" - loss etc.
    @abstractmethod
    def update_with_experience(self, **kwargs) -> dict: pass

    @abstractmethod
    # returns Actor TOP save directory
    def _get_save_topdir(self) -> str: pass

    # returns Actor save directory
    def get_save_dir(self) -> str:
        return f'{self._get_save_topdir()}/{self.name}'

    # saves (self) Actor state
    @abstractmethod
    def save(self): pass

    # returns some info about Actor
    @abstractmethod
    def __str__(self) -> str: pass