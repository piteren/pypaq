"""

 2022 (c) piteren

    Policy Gradients Trainer

"""
import numpy as np

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
            **kwargs):

        FATrainer.__init__(self, actor=actor, **kwargs)
        self.actor = actor # INFO: just type "upgrade" for pycharm editor
        self.discount = discount
        self.use_mavg = use_mavg
        self.movavg_factor = mavg_factor
        self.do_zscore = do_zscore

        self._rlog.info('*** PGTrainer *** initialized')
        self._rlog.info(f'> discount: {self.discount}')

    # PGActor update method
    def _update_actor(self, inspect=False) -> dict:

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

        if inspect:

            obs_arr = np.asarray(observations)
            oL = np.split(obs_arr, obs_arr.shape[-1], axis=-1)

            two_dim_multi(
                ys=     oL,
                names=  [f'obs_{ix}' for ix in range(len(oL))])

            # prepare all 4 for inspect
            ret_mavg = []
            ret_disc = []
            for rs in episode_rewards:
                ret_mavg += movavg_return(rewards=rs, factor=self.movavg_factor)
                ret_disc += discounted_return(rewards=rs, discount=self.discount)
            ret_mavg_norm = zscore_norm(ret_mavg)
            ret_disc_norm = zscore_norm(ret_disc)

            two_dim_multi(
                ys=     [
                    rewards,
                    ret_mavg,
                    ret_disc,
                    ret_mavg_norm,
                    ret_disc_norm],
                names=  [
                    'rewards',
                    'ret_mavg',
                    'ret_disc',
                    'ret_mavg_norm',
                    'ret_disc_norm'],
                legend_loc= 'lower left')

        dreturns = []
        if self.use_mavg:
            for rs in episode_rewards:
                dreturns += movavg_return(rewards=rs, factor=self.movavg_factor)
        else:
            for rs in episode_rewards:
                dreturns += discounted_return(rewards=rs, discount=self.discount)
        if self.do_zscore: dreturns = zscore_norm(dreturns)
        dreturns = np.asarray(dreturns, dtype=np.float32)

        out = self.actor.update_with_experience(
            observations=   observations,
            actions=        actions,
            dreturns=       dreturns,
            inspect=        inspect)

        value = out.pop('value').cpu().detach().numpy()
        advantage = out.pop('advantage').cpu().detach().numpy()

        if inspect:
            two_dim_multi(
                ys=     [
                    dreturns,
                    value,
                    advantage,
                ],
                names=  [
                    'dreturns',
                    'value',
                    'advantage',
                ],
                legend_loc= 'lower left')

        return out