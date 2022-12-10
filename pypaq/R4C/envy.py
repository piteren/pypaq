"""

 2022 (c) piteren

    Envy is a base environment interface

    RLEnvy is an interface that defines base RL methods used by Actor or Trainer (..or other objects)

        RLEnvy returns reward after step (run). This value may (should?) be processed / overridden by Trainer.
        Trainer is supposed to train Actor using information of reward that he defines / corrects observing an Envy.
        He may apply discount, factor, moving average etc to those values.
        (Actor does not need reward to act with policy.)

"""

from abc import abstractmethod, ABC
import numpy as np
from typing import List, Tuple, Optional

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.helpers import RLException



# base Environment interface
class Envy(ABC):

    def __init__(
            self,
            seed: int,
            logger=     None,
            loglevel=   20):
        self._log = logger or get_pylogger(level=loglevel)
        self.seed = seed
        self._log.info(f'*** Envy *** initialized')
        self._log.info(f'> seed:      {self.seed}')
        self._log.info(f'> max steps: {self.get_max_steps()}')

    # plays action, goes to new state
    @abstractmethod
    def run(self, action: object) -> None: pass

    # returns observation of current state
    @abstractmethod
    def get_observation(self) -> object: pass

    # resets Envy (self) to initial state with given seed
    @abstractmethod
    def _reset_with_seed(self, seed: int): pass

    # resets Envy (self) to initial state
    def reset(self):
        self._reset_with_seed(seed=self.seed)
        self.seed += 1

    # returns max number of steps in one episode, None means infinite
    @abstractmethod
    def get_max_steps(self) -> Optional[int]: pass


# adds to Envy methods needed by base RL algorithms
class RLEnvy(Envy, ABC):

    def run(self, action: object) -> Tuple[
        float,  # reward
        bool,   # is terminal
        bool    # has won
    ]: pass

    # Envy current state rendering (for debug, preview etc.)
    def render(self): pass

    # prepares numpy vector from observation, it may be implemented by RLEnvy, but is not mandatory, otherwise Actor should implement on itself
    def prep_observation_vec(self, observation: object) -> np.ndarray:
        raise RLException('RLEnvy not implemented prep_observation_vec()')


# interface of RL Environment with finite actions number
class FiniteActionsRLEnvy(RLEnvy):

    def __init__(self, **kwargs):
        RLEnvy.__init__(self, **kwargs)
        self._log.info(f'*** FiniteActionsRLEnvy *** initialized')
        self._log.info(f'> num_actions: {self.num_actions()}')

    # returns list of valid actions
    @abstractmethod
    def get_valid_actions(self) -> List[object]: pass

    # returns number of Envy actions
    def num_actions(self) -> int:
        return len(self.get_valid_actions())