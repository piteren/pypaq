"""

 2022 (c) piteren

    Policy Gradients Trainer

"""

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.plots import two_dim_multi
from pypaq.R4C.helpers import zscore_norm, extract_from_batch, discounted_return, movavg_return
from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.trainer import FATrainer


class PGTrainer(FATrainer):

    def __init__(
            self,
            actor: PGActor,
            discount: float,    # for discounted returns
            use_mavg: bool,     # use movavg to calculate discounted returns
            mavg_factor: float,
            do_zscore: bool,
            logger=     None,
            loglevel=   20,
            **kwargs):

        if not logger:
            logger = get_pylogger(
                name=       'PGTrainer',
                add_stamp=  True,
                folder=     None,
                level=      loglevel)
        self.__log = logger
        self.__log.info(f'*** PGTrainer initializes, discount: {discount}')

        FATrainer.__init__(
            self,
            actor=  actor,
            logger= self.__log,
            **kwargs)
        self.actor = actor # INFO: just type "upgrade" for pycharm editor
        self.discount = discount
        self.use_mavg = use_mavg
        self.movavg_factor = mavg_factor
        self.do_zscore = do_zscore

    # PGActor update method
    def _update_actor(self, inspect=False) -> float:

        # get all and flush
        batch = self.memory.get_all()
        self.memory.clear()

        observations = extract_from_batch(batch, 'observation')
        actions =      extract_from_batch(batch, 'action')
        rewards =      extract_from_batch(batch, 'reward')
        terminals =    extract_from_batch(batch, 'terminal')

        # ********************************************************************************************* prepare dreturns

        # split rewards into episodes
        episode_rewards = []
        cep = []
        for r,t in zip(rewards,terminals):
            cep.append(r)
            if t:
                episode_rewards.append(cep)
                cep = []
        if cep: episode_rewards.append(cep)

        # prepare both for inspect
        dreturns_mavg = []
        dreturns_disc = []
        for rs in episode_rewards:
            dreturns_mavg += movavg_return(rewards=rs, factor=self.movavg_factor)
            dreturns_disc += discounted_return(rewards=rs, discount=self.discount)

        dreturns = dreturns_mavg if self.use_mavg else dreturns_disc
        dreturns_norm = zscore_norm(dreturns)

        if inspect:
            two_dim_multi(
                ys=         [rewards, dreturns_mavg, dreturns_disc, dreturns_norm],
                names=      ['rewards', 'dreturns_mavg', 'dreturns_disc', 'dreturns_norm'],
                legend_loc= 'lower left')

        return self.actor.update_with_experience(
            observations=   observations,
            actions=        actions,
            dreturns=       dreturns_norm if self.do_zscore else dreturns,
            inspect=        inspect)