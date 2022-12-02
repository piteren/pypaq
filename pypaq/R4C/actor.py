"""

 2022 (c) piteren

    RL Actor

"""

from abc import abstractmethod, ABC
import numpy as np

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.envy import RLEnvy



class Actor(ABC):

    # returns Actor action based on observation according to Actor policy, optionally action may be sampled from probability
    @abstractmethod
    def get_policy_action(self, observation: object) -> object: pass


class TrainableActor(Actor, ABC):

    def __init__(
            self,
            envy: RLEnvy,
            seed: int,
            logger=     None,
            loglevel=   20):

        self._log = logger or get_pylogger(level=loglevel)
        self._envy = envy
        self.seed = seed

        self._log.info('*** TrainableActor *** initialized')
        self._log.info(f'> Envy: {self._envy.__class__.__name__}')
        self._log.info(f'> seed: {self.seed}')

    # prepares numpy vector from observation, it is a private / internal skill of Actor, to be implemented for each Envy
    @abstractmethod
    def _get_observation_vec(self, observation: object) -> np.ndarray: pass

    # add sampling (from probability?) option which may be helpful for training
    @abstractmethod
    def get_policy_action(self, observation:object, sampled=False) -> object: pass

    # updates self with (batch of) experience data (given with kwargs), returns Actor "metric" - loss etc. (float)
    @abstractmethod
    def update_with_experience(self, **kwargs) -> float: pass

    # saves (self) Actor state
    @abstractmethod
    def save(self) -> None: pass

    # returns some info about Actor
    @abstractmethod
    def __str__(self) -> str: pass