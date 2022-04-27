"""

 2022 (c) piteren

    RL Actor

"""

from abc import abstractmethod, ABC


class Actor(ABC):

    # returns Actor action
    @abstractmethod
    def get_policy_action(self, observation, sampled=False):
        """
        returns Actor action based on observation according to Actor policy,
        optionally action may be sampled from probability (for probabilistic policy)

        """
        pass

    # updates self with batch of experience data
    @abstractmethod
    def update_batch(self, **kwargs): pass