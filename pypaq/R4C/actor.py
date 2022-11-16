"""

 2022 (c) piteren

    RL Actor

"""

from abc import abstractmethod, ABC


class Actor(ABC):

    # returns Actor action based on observation according to Actor policy, optionally action may be sampled from probability
    @abstractmethod
    def get_policy_action(self, observation: object) -> object: pass


class TrainableActor(Actor, ABC):

    # add sampling option which may be helpful for training
    @abstractmethod
    def get_policy_action(self, observation:object, sampled=False) -> object: pass

    # updates self with (batch of) experience data (given with kwargs), returns Actor "metric" - loss etc. (float)
    @abstractmethod
    def update_with_experience(self, **kwargs) -> float: pass

    # returns some info about Actor
    @abstractmethod
    def __str__(self): pass