from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.qlearning.qlearning_trainer import QLearningTrainer
from pypaq.R4C.qlearning.qtable.qtable_actor import QTableActor


class QTableTrainer(QLearningTrainer):

    def __init__(
            self,
            actor: QTableActor,
            update_rate,
            logger=     None,
            loglevel=   20,
            **kwargs):

        if not logger:
            logger = get_pylogger(
                name=       'QTableTrainer',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.__log = logger
        self.__log.info(f'*** QTableTrainer initializes, actor: {actor.__class__.__name__}, update_rate: {update_rate}')

        self.actor = actor
        self.actor.set_update_rate(update_rate)
        QLearningTrainer.__init__(
            self,
            actor=      self.actor,
            logger=     self.__log,
            **kwargs)