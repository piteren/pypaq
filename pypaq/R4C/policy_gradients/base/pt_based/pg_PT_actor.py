"""

 2021 (c) piteren

    PolicyGradients NN Actor, TF based

    TODO: implement parallel training, in batches (many envys)

"""

from abc import ABC
import numpy as np
from typing import Optional

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.policy_gradients.pg_actor import PG_Actor
from pypaq.R4C.policy_gradients.base.pt_based.pg_PT_module import PGModel
from pypaq.torchness.motorch import MOTorch, Module



class PG_PTActor(PG_Actor, ABC):

    def __init__(
            self,
            nngraph: Optional[type(Module)]=    PGModel,
            logger=                             None,
            loglevel=                           20,
            **kwargs):

        logger_given = bool(logger)
        if not logger_given:
            logger = get_pylogger(
                name=       'PG_PTActor',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.__log = logger

        PG_Actor.__init__(
            self,
            nnwrap=     MOTorch,
            nngraph=    nngraph,
            logger=     self.__log if logger_given else None, # if user gives logger we assume it to be nice logger, otherwise we want to pas None up to NNWrap, which manages logger in pretty way
            loglevel=   loglevel,
            **kwargs)

    def get_policy_probs(self, observation: object) -> np.ndarray:
        obs_vec = self._get_observation_vec(observation)
        return self.nnw(obs_vec)['probs'].detach().cpu().numpy()

    # optimized with batch call to NN
    def get_policy_probs_batch(self, observations) -> np.ndarray:
        obs_vecs = self._get_observation_vec_batch(observations)
        return self.nnw(obs_vecs)['probs'].detach().cpu().numpy()

    # updates self NN with batch of data
    def update_with_experience(
            self,
            observations,
            actions,
            dreturns,
            inspect=    False) -> float:

        obs_vecs = self._get_observation_vec_batch(observations)

        out = self.nnw.backward(obs_vecs, actions, dreturns)

        self._upd_step += 1

        loss = float(out['loss'])
        gn = float(out['gg_norm'])
        gn_avt = float(out['gg_avt_norm'])
        amax_prob = float(out['amax_prob'])
        amin_prob = float(out['amin_prob'])
        ace = float(out['actor_ce_mean'])
        cLR = float(out['currentLR'])

        self.nnw.log_TB(loss,       'upd/loss',             step=self._upd_step)
        self.nnw.log_TB(gn,         'upd/gn',               step=self._upd_step)
        self.nnw.log_TB(gn_avt,     'upd/gn_avt',           step=self._upd_step)
        self.nnw.log_TB(amax_prob,  'upd/amax_prob',        step=self._upd_step)
        self.nnw.log_TB(amin_prob,  'upd/amin_prob',        step=self._upd_step)
        self.nnw.log_TB(ace,        'upd/actor_ce_mean',    step=self._upd_step)
        self.nnw.log_TB(cLR,        'upd/cLR',              step=self._upd_step)

        return loss