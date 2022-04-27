"""

 2022 (c) piteren

    Reinforcement Environment Interfaces

    Envy is a base environment interface

    RLEnvy is an interface that defines base RL methods used by Actor or Trainer or other objects

        RLEnvy may implement get_last_action_reward() and get_reward() methods (reward function)
        that in fact should be implemented by a Trainer.
        (Actor does not need reward to act with policy, Trainer is supposed to train Actor
        using information of reward that he defines observing an Envy)
        get_reward methods may be implemented by RLEnvy as an generic baseline used by Trainer,
        he always may override it with custom reward function.

"""

from abc import abstractmethod, ABC


# base Environment interface
class Envy(ABC):

    def __init__(self, name:str):
        self.name = name

    # resets envy to initial state
    @abstractmethod
    def reset(self): pass

    # returns observation of current state
    @abstractmethod
    def get_observation(self) -> object: pass

    # plays action, goes to new state
    @abstractmethod
    def run(self, action): pass


# wraps Envy with methods needed by base RL algorithms
class RLEnvy(Envy, ABC):

    # returns reward of last action played by envy, this is in fact Trainer function, but it is easier to implement it with an Envy
    @abstractmethod
    def get_last_action_reward(self) -> float: pass

    # returns reward based on observations and action, this is in fact Trainer function, but it is easier to implement it with an Envy
    @abstractmethod
    def get_reward(self, prev_observation, action, next_observation) -> float: pass

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

    # returns number of envy actions
    @abstractmethod
    def num_actions(self) -> int: pass