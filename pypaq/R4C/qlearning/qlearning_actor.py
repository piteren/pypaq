"""

 2022 (c) piteren

    QLearningActor baseline (abstract interface)

"""

from abc import abstractmethod, ABC
import numpy as np

from pypaq.R4C.actor import Actor


class QLearningActor(Actor, ABC):

    # returns QVs (list or any other iterable for all actions) for given observation
    @abstractmethod
    def get_QVs(self, observation) -> np.ndarray: pass

    # returns QVs (for all actions) for given observations batch, may be overridden with optimized version
    def get_QVs_batch(self, observations):
        return [self.get_QVs(observation) for observation in observations]

    def get_policy_action(self, observation, sampled=False) -> int:
        assert not sampled, 'ERR: sampled action is not supported for QLActor'
        return int(np.argmax(self.get_QVs(observation)))

    # updates QV for given observation and action (single observation/action)
    @abstractmethod
    def upd_QV(
            self,
            observation,
            action,
            new_qv): pass

    # updates QV for given observations and actions batch, may be overridden with optimized version
    def update_batch(
            self,
            observations,
            actions,
            new_qvs):
        for ob,ac,nq in zip(observations,actions,new_qvs):
            self.upd_QV(ob,ac,nq)