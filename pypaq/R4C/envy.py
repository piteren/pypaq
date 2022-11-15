"""

 2022 (c) piteren

    Envy is a base environment interface

    RLEnvy is an interface that defines base RL methods used by Actor or Trainer (..or other objects)

        RLEnvy implements get_reward methods: get_last_action_reward() and get_reward().
        Values returned with those methods always may (should?) be overridden by Trainer.
        Trainer is supposed to train Actor using information of reward that he defines / corrects observing an Envy.
        He may apply discount, factor, moving average etc to those values.
        (Actor does not need reward to act with policy.)

"""

from abc import abstractmethod, ABC
from typing import List

from pypaq.lipytools.pylogger import get_pylogger


# base Environment interface
class Envy(ABC):

    def __init__(self, name:str, logger):
        self.name = name
        if not logger: logger = get_pylogger(name=self.name)
        self.__log = logger
        self.__log.debug(f'*** Envy {self.name} initialized!')

    # resets Envy (self) to initial state
    @abstractmethod
    def reset(self): pass

    # returns observation of current state
    @abstractmethod
    def get_observation(self) -> object: pass

    # plays action, goes to new state
    @abstractmethod
    def run(self, action: object): pass


# adds to Envy methods needed by base RL algorithms
class RLEnvy(Envy, ABC):

    # returns reward of last action played by envy, this is in fact Trainer function, but it is easier to implement it with an Envy
    @abstractmethod
    def get_last_action_reward(self) -> float: pass

    # returns reward based on observations and action, this is in fact Trainer function, but it is easier to implement it with an Envy
    @abstractmethod
    def get_reward(
            self,
            prev_observation: object,
            action: object,
            next_observation: object) -> float: pass

    # returns True if episode finished and has been won, for some Envies it wont return True whenever
    @abstractmethod
    def won_episode(self) -> bool: pass

    # returns True if episode finished and has been lost, for some Envies it wont return True whenever
    @abstractmethod
    def lost_episode(self) -> bool: pass

    # returns True if is in terminal state (won or lost >> episode finished)
    def is_terminal(self) -> bool:
        return self.lost_episode() or self.won_episode()

    # Envy rendering (for debug, preview etc.)
    def render(self): pass


# interface of RL Environment with finite actions number
class FiniteActionsRLEnvy(RLEnvy):

    # returns number of Envy actions
    @abstractmethod
    def num_actions(self) -> int: pass

    # returns list of valid actions
    @abstractmethod
    def get_valid_actions(self) -> List[object]: pass