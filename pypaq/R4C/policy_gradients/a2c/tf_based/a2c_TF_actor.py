from abc import ABC
from typing import Optional, Callable

from pypaq.R4C.policy_gradients.base.tf_based.pg_TF_actor import PG_TFActor
from pypaq.R4C.policy_gradients.a2c.tf_based.a2c_TF_graph import a2c_graph


class A2C_TFActor(PG_TFActor, ABC):

    def __init__(self, nngraph:Optional[Callable]=a2c_graph, **kwargs):
        # TODO: wrong logger name, see pg_TF_actor #33
        PG_TFActor.__init__(self, nngraph=nngraph, **kwargs)

    # overrides PG_TFActor with more log_TB
    def update_with_experience(
            self,
            observations,
            actions,
            dreturns,
            inspect=    False) -> float:

        obs_vecs = self._get_observation_vec_batch(observations)
        _, loss, loss_actor, loss_critic, gn, gn_avt, amax_prob, amin_prob, ace = self.nnw.backward(
            feed_dict=  {
                self.nnw['observation_PH']:  obs_vecs,
                self.nnw['action_PH']:       actions,
                self.nnw['return_PH']:       dreturns},
            fetches=    [
                self.nnw['optimizer'],
                self.nnw['loss'],
                self.nnw['loss_actor'],
                self.nnw['loss_critic'],
                self.nnw['gg_norm'],
                self.nnw['gg_avt_norm'],
                self.nnw['amax_prob'],
                self.nnw['amin_prob'],
                self.nnw['actor_ce_mean']])

        self._upd_step += 1

        self.nnw.log_TB(loss,        'upd/loss',             step=self._upd_step)
        self.nnw.log_TB(loss_actor,  'upd/loss_actor',       step=self._upd_step)
        self.nnw.log_TB(loss_critic, 'upd/loss_critic',      step=self._upd_step)
        self.nnw.log_TB(gn,          'upd/gn',               step=self._upd_step)
        self.nnw.log_TB(gn_avt,      'upd/gn_avt',           step=self._upd_step)
        self.nnw.log_TB(amax_prob,   'upd/amax_prob',        step=self._upd_step)
        self.nnw.log_TB(amin_prob,   'upd/amin_prob',        step=self._upd_step)
        self.nnw.log_TB(ace,         'upd/actor_ce_mean',    step=self._upd_step)

        return loss_actor