from abc import ABC
import numpy as np
from typing import Optional, List, Dict, Any

from pypaq.R4C.helpers import extract_from_batch
from pypaq.R4C.actor import TrainableActor
from pypaq.R4C.envy import FiniteActionsRLEnvy
from pypaq.R4C.policy_gradients.pg_actor_module import PGActorModule
from pypaq.torchness.motorch import MOTorch, Module

# TODO: implement parallel training, in batches (many envys)

# PolicyGradients TrainableActor, NN based
class PGActor(TrainableActor, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            name: str=                              'PGActor',
            module_type: Optional[type(Module)]=    PGActorModule,
            seed: int=                              123,
            **kwargs):

        TrainableActor.__init__(
            self,
            envy=   envy,
            name=   name,
            **kwargs)
        self._envy = envy  # to update type (for pycharm only)

        np.random.seed(seed)

        # some overrides and updates
        if 'logger' in kwargs: kwargs.pop('logger')     # NNWrap will always create own logger (since then it is not given) with optionally given level
        kwargs['num_actions'] = self._envy.num_actions()
        kwargs['observation_width'] = self._get_observation_vec(self._envy.get_observation()).shape[-1]

        self.model = MOTorch(
            module_type=    module_type,
            name=           self.name,
            seed=           seed,
            **kwargs)

        self._rlog.info(f'*** PGActor : {self.name} *** (NN based) initialized')

    # vectorization of observations batch, may be overridden with more optimal custom implementation
    def _get_observation_vec_batch(self, observations: List[object]) -> np.ndarray:
        return np.asarray([self._get_observation_vec(v) for v in observations])

    def get_policy_probs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        return self.model(obs_vec)['probs'].detach().cpu().numpy()

    # batch call to NN
    def get_policy_probs_batch(self, observations: List[object]) -> np.ndarray:
        obs_vecs = self._get_observation_vec_batch(observations)
        return self.model(obs_vecs)['probs'].detach().cpu().numpy()

    # gets policy action based on policy (action) probs
    def get_policy_action(self, observation: object, sampled=False) -> int:
        probs = self.get_policy_probs(observation)
        if sampled: action = np.random.choice(self._envy.num_actions(), p=probs)
        else:       action = np.argmax(probs)
        return int(action)

    def get_policy_action_batch(self, observations: List[object], sampled=False) -> np.ndarray:
        probs = self.get_policy_probs_batch(observations)
        if sampled: actions = np.random.choice(self._envy.num_actions(), size=probs.shape[-1], p=probs)
        else:       actions = np.argmax(probs, axis=-1)
        return actions

    # updates self NN with batch of data
    def update_with_experience(
            self,
            batch: List[Dict[str,Any]],
            inspect: bool,
    ) -> Dict[str, Any]:

        observations = extract_from_batch(batch, 'observation')

        obs_vecs = self._get_observation_vec_batch(observations)
        out = self.model.backward(
            observation=    obs_vecs,
            action_taken=   extract_from_batch(batch, 'action'),
            dreturn=        extract_from_batch(batch, 'dreturn'))

        out.pop('logits')
        if 'probs' in out: out['probs'] = out['probs'].cpu().detach().numpy()

        return out

    def _get_save_topdir(self) -> str:
        return self.model['save_topdir']

    def save(self):
        self.model.save()

    def __str__(self):
        return self.model.__str__()