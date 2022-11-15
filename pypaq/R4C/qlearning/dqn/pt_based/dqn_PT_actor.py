"""

 2022 (c) piteren

    DQNPTActor - DQNActor implemented with MOTorch (PyTorch)

"""

from abc import ABC
import numpy as np

from pypaq.lipytools.pylogger import get_pylogger, get_hi_child
from pypaq.lipytools.little_methods import stamp
from pypaq.R4C.qlearning.qlearning_actor import QLearningActor
from pypaq.R4C.qlearning.dqn.pt_based.dqn_PT_module import LinModel
from pypaq.torchness.motorch import MOTorch


# DQN PyTorch (NN) based QLearningActor
class DQN_PTActor(QLearningActor, ABC):

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

        if 'name' not in mdict: mdict['name'] = f'dqnPT_{stamp()}'
        mdict['num_actions'] = num_actions
        mdict['observation_width'] = self.get_observation_vec(observation).shape[-1]
        mdict['save_topdir'] = save_topdir

        self.__log.info(f'*** DQN_PTActor {mdict["name"]} (PyTorch based) initializes..')
        self.__log.info(f'> num_actions:       {mdict["num_actions"]}')
        self.__log.info(f'> observation_width: {mdict["observation_width"]}')
        self.__log.info(f'> save_topdir:       {mdict["save_topdir"]}')

        self.nn = MOTorch(
            module=     module,
            logger=     None if not logger_given else get_hi_child(self.__log, 'MOTorch', higher_level=False),
            loglevel=   loglevel,
            **mdict)

        self._upd_step = 0

        self.__log.info(f'DQN_PTActor initialized')

    def get_QVs(self, observation: np.ndarray) -> np.ndarray:
        return self.nn(observation)['logits'].detach().cpu().numpy()

    # optimized with single call with a batch of observations
    def get_QVs_batch(self, observations: np.ndarray) -> np.ndarray:
        return self.nn(observations)['logits'].detach().cpu().numpy()

    # optimized with single call to session with a batch of data
    def update_with_experience(
            self,
            observations: np.ndarray,
            actions: np.ndarray,
            new_qvs: np.ndarray,
            inspect=    False) -> float:

        full_qvs = np.zeros_like(observations)
        mask = np.zeros_like(observations)
        for v,pos in zip(new_qvs, enumerate(actions)):
            full_qvs[pos] = v
            mask[pos] = 1

        self.__log.log(5, f'>>> observations.shape, actions.shape, new_qvs.shape: {observations.shape}, {actions.shape}, {new_qvs.shape}')
        self.__log.log(5, f'>>> actions: {actions}')
        self.__log.log(5, f'>>> new_qvs: {new_qvs}')
        self.__log.log(5, f'>>> full_qvs: {full_qvs}')
        self.__log.log(5, f'>>> mask: {mask}')

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

    # INFO: not used since this Actor updates only with batches
    def upd_QV(
            self,
            observation: np.ndarray,
            action: int,
            new_qv: float) -> float:
        raise NotImplementedError

    def save(self): self.nn.save()

    def __str__(self): return self.nn.__str__()