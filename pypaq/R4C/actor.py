"""

 2022 (c) piteren

    RL Actor

"""

from abc import abstractmethod, ABC
import numpy as np


class Actor(ABC):

    # prepares np.ndarray from observation
    @abstractmethod
    def get_observation_vec(self, observation: object) -> np.ndarray: pass

    # returns Actor action based on observation according to Actor policy, optionally action may be sampled from probability
    @abstractmethod
    def get_policy_action(
            self,
            observation: object,
            sampled=    False) -> object: pass

    # updates self with (batch of) experience data, returns Actor "metric" - loss etc. (float)
    @abstractmethod
    def update_with_experience(self, **kwargs) -> float: pass