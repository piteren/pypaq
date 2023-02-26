from abc import abstractmethod, ABC
import numpy as np
from typing import List, Dict, Any

from pypaq.R4C.helpers import extract_from_batch
from pypaq.R4C.actor import TrainableActor
from pypaq.R4C.envy import FiniteActionsRLEnvy
from pypaq.lipytools.softmax import softmax


# QLearningActor, supports finite actions space environments (FiniteActionsRLEnvy)
class QLearningActor(TrainableActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            seed: int,
            **kwargs):

        TrainableActor.__init__(self, envy=envy, **kwargs)
        self._envy = envy  # to update type (for pycharm only)

        np.random.seed(seed)

        self._rlog.info('*** QLearningActor *** initialized')
        self._rlog.info(f'> num_actions: {self._envy.num_actions()}')
        self._rlog.info(f'> seed:        {seed}')

    # returns QVs (QV for all actions) for given observation
    @abstractmethod
    def _get_QVs(self, observation: object) -> np.ndarray: pass

    # returns QVs (for all actions) for given observations batch, here baseline implementation - may be overridden with optimized version
    def get_QVs_batch(self, observations: List[object]) -> np.ndarray:
        return np.asarray([self._get_QVs(o) for o in observations])

    def get_policy_action(
            self,
            observation: object,
            sampled=    False) -> int:

        qvs = self._get_QVs(observation)

        if sampled:
            obs_probs = softmax(qvs) # baseline with softmax on QVs
            return np.random.choice(len(qvs), p=obs_probs)

        return int(np.argmax(qvs))

    # updates QV for given observation and action (single observation/action), returns Actor "metric" - loss etc. (float)
    @abstractmethod
    def _upd_QV(
            self,
            observation: object,
            action: int,
            new_qv: float) -> float: pass

    # updates QV
    def update_with_experience(
            self,
            batch: List[Dict[str, Any]],
            inspect: bool,
    ) -> Dict[str, Any]:

        loss = 0.0

        observations = extract_from_batch(batch, 'observation')
        actions = extract_from_batch(batch, 'action')
        new_qvs = extract_from_batch(batch, 'new_qvs')

        for ob, ac, nq in zip(observations, actions, new_qvs):
            loss += self._upd_QV(
                observation=    ob,
                action=         ac,
                new_qv=         nq)

        return {'loss': loss}