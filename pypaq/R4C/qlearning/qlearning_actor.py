"""

 2022 (c) piteren

    QLearningActor interface

"""

from abc import abstractmethod, ABC
import numpy as np

from pypaq.R4C.actor import Actor
from pypaq.lipytools.softmax import softmax



class QLearningActor(Actor, ABC):

    # returns QVs (QV for all actions) for given observation
    @abstractmethod
    def get_QVs(self, observation: np.ndarray) -> np.ndarray: pass

    # returns QVs (for all actions) for given observations batch, here baseline implementation - may be overridden with optimized version
    def get_QVs_batch(self, observations: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(
            func1d= self.get_QVs,
            axis=   -1,
            arr=    observations)

    def get_policy_action(
            self,
            observation: np.ndarray,
            sampled=    False) -> int:

        qvs = self.get_QVs(observation)

        if sampled:
            obs_probs = softmax(qvs) # baseline with softmax on QVs
            return np.random.choice(len(observation), p=obs_probs)

        return int(np.argmax(qvs))

    # updates QV for given observation and action (single observation/action), returns Actor "metric" - loss etc. (float)
    @abstractmethod
    def upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float: pass

    # updates QV for given observations and actions batch, may be overridden with optimized version
    def update_with_experience(
            self,
            observations: np.ndarray,   # batch of observations
            actions: np.ndarray,        # batch of selected actions for given observations (do not have to come from Actor policy!)
            new_qvs: np.ndarray,        # batch of QV to be updated (QV for selected action only)
    ) -> float:
        loss = 0.0
        for ob, ac, nq in zip(observations, actions, new_qvs):
            loss += self.upd_QV(
                observation=    ob,
                action=         int(ac),
                new_qv=         float(nq))
        return loss