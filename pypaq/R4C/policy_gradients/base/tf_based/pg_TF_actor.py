"""

 2021 (c) piteren

    PolicyGradients NN Actor, TF based

"""

from abc import ABC
import numpy as np
from typing import Optional, Callable, List

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.policy_gradients.base.tf_based.pg_TF_graph import pga_graph
from pypaq.neuralmess.nemodel import NEModel


class PG_TFActor(PGActor, ABC):

    def __init__(self, nngraph:Optional[Callable]=pga_graph, **kwargs):
        PGActor.__init__(self, nnwrap=NEModel, nngraph=nngraph, **kwargs)

    def get_policy_probs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        probs = self.nnw(
            feed_dict=  {self.nnw['observation_PH']: [obs_vec]},
            fetches=    self.nnw['action_prob'])
        return probs[0]

    # optimized with batch call to NN
    def get_policy_probs_batch(self, observations: List[object]) -> np.ndarray:
        obs_vecs = self._get_observation_vec_batch(observations)
        probs = self.nnw(
            feed_dict=  {self.nnw['observation_PH']: obs_vecs},
            fetches=    self.nnw['action_prob'])
        return probs

    # updates self NN with batch of data
    def update_with_experience(
            self,
            observations,
            actions,
            dreturns,
            inspect=    False) -> float:

        obs_vecs = self._get_observation_vec_batch(observations)
        _, loss, gn, gn_avt, amax_prob, amin_prob, ace = self.nnw.backward(
            feed_dict=  {
                self.nnw['observation_PH']:  obs_vecs,
                self.nnw['action_PH']:       actions,
                self.nnw['return_PH']:       dreturns},
            fetches=    [
                self.nnw['optimizer'],
                self.nnw['loss'],
                self.nnw['gg_norm'],
                self.nnw['gg_avt_norm'],
                self.nnw['amax_prob'],
                self.nnw['amin_prob'],
                self.nnw['actor_ce_mean']])

        self._upd_step += 1

        self.nnw.log_TB(loss,        'upd/loss',             step=self._upd_step)
        self.nnw.log_TB(gn,          'upd/gn',               step=self._upd_step)
        self.nnw.log_TB(gn_avt,      'upd/gn_avt',           step=self._upd_step)
        self.nnw.log_TB(amax_prob,   'upd/amax_prob',        step=self._upd_step)
        self.nnw.log_TB(amin_prob,   'upd/amin_prob',        step=self._upd_step)
        self.nnw.log_TB(ace,         'upd/actor_ce_mean',    step=self._upd_step)

        return loss