"""

 2022 (c) piteren

    DQNPTActor - DQNActor implemented with MOTorch (PyTorch)

"""

from abc import ABC
import numpy as np

from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.R4C.qlearning.dqn.dqn_actor_common import DQNActor
from pypaq.R4C.qlearning.dqn.pt_based.dqn_PT_module import LinModel
from pypaq.torchness.motorch import MOTorch


# DQN PyTorch (NN) based QLearningActor
class DQN_PTActor(DQNActor, ABC):

    def __init__(
            self,
            num_actions: int,
            observation,
            mdict: dict,
            module=         LinModel,
            save_topdir=    '_models',
            logger=         None,
            loglevel=       20):

        logger_given = bool(logger)
        if not logger_given:
            logger = get_pylogger(
                name=       'DQN_PTActor',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.__log = logger

        mdict['save_topdir'] = save_topdir
        self.__log.info(f'DQN (PyTorch based) Actor initializes, save_topdir: {mdict["save_topdir"]}')

        DQNActor.__init__(
            self,
            num_actions=    num_actions,
            observation=    observation,
            mdict=          mdict,
            logger=         self.__log,
            name_pfx=       'dqnPT')

        self.nn = self._init_model(
            module=         module,
            mdict=          mdict,
            logger=         None if not logger_given else get_hi_child(self.__log, 'MOTorch', higher_level=False),
            loglevel=       loglevel)

        self._upd_step = 0

        self.__log.info(f'DQN (PyTorch based) Actor initialized')

    def _init_model(self, module, mdict, logger, loglevel):
        return MOTorch(
            module=         module,
            logger=         logger,
            loglevel=       loglevel,
            **mdict)

    def get_QVs(self, observation) -> np.ndarray:
        return self.nn(observation)['logits'].detach().cpu().numpy()

    # optimized with single call with a batch of observations
    def get_QVs_batch(self, observations) -> np.ndarray:
        return self.nn(observations)['logits'].detach().cpu().numpy()

    # optimized with single call to session with a batch of data
    def update_with_experience(
            self,
            observations: np.ndarray,
            actions: np.ndarray,
            new_qvs: np.ndarray) -> float:

        full_qvs = np.zeros_like(observations)
        mask = np.zeros_like(observations)
        for v,pos in zip(new_qvs, enumerate(actions)):
            full_qvs[pos] = v
            mask[pos] = 1

        """
        print(observations.shape, actions.shape, new_qvs.shape)
        print(actions)
        print(new_qvs)
        print(full_qvs)
        print(mask)
        """

        out = self.nn.backward(observations, full_qvs, mask)

        self._upd_step += 1

        loss = float(out['loss'])
        gn = float(out['gg_norm'])
        gn_avt = float(out['gg_avt_norm'])
        cLR = float(out['currentLR'])

        self.nn.log_TB(loss,    'upd/loss',     step=self._upd_step)
        self.nn.log_TB(gn,      'upd/gn',       step=self._upd_step)
        self.nn.log_TB(gn_avt,  'upd/gn_avt',   step=self._upd_step)
        self.nn.log_TB(cLR,     'upd/cLR',      step=self._upd_step)

        return loss