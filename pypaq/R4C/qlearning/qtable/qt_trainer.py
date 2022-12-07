"""

 2022 (c) piteren

    QTableTrainer - Trainer specific for training of QTableActor, sets update_rate of Actor.

"""

from pypaq.R4C.qlearning.ql_trainer import QLearningTrainer
from pypaq.R4C.qlearning.qtable.qt_actor import QTableActor


class QTableTrainer(QLearningTrainer):

    def __init__(
            self,
            actor: QTableActor,
            update_rate: float,
            **kwargs):

        QLearningTrainer.__init__(self, actor=actor, **kwargs)
        self.actor = actor
        self.actor.set_update_rate(update_rate)

        self._rlog.info('*** QTableTrainer *** initialized')
        self._rlog.info('> actor update_rate: {update_rate}')