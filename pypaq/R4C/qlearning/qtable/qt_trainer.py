"""

 2022 (c) piteren

    QTableTrainer - Trainer specific for training of QTableActor, sets update_rate of Actor.

"""

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.qlearning.ql_trainer import QLearningTrainer
from pypaq.R4C.qlearning.qtable.qt_actor import QTableActor


class QTableTrainer(QLearningTrainer):

    def __init__(
            self,
            actor: QTableActor,
            update_rate: float,
            logger=     None,
            loglevel=   20,
            **kwargs):

        if not logger:
            logger = get_pylogger(
                name=       'QTableTrainer',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self._log = logger
        self._log.info(f'*** QTableTrainer initializes, actor: {actor.__class__.__name__}, update_rate: {update_rate}')

        self.actor = actor
        self.actor.set_update_rate(update_rate)
        QLearningTrainer.__init__(
            self,
            actor=      self.actor,
            logger=     self._log,
            **kwargs)