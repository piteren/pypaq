import numpy as np

from pypaq.R4C.policy_gradients.actor_critic_shared.ac_shared_model import ACSharedModel
from pypaq.R4C.trainer import FATrainer


class ACSharedTrainer(FATrainer):

    def __init__(
            self,
            acs_model: ACSharedModel,
            verb=           1,
            **kwargs):

        FATrainer.__init__(self, actor=acs_model, verb=verb, **kwargs)
        self.actor = acs_model # INFO: type "upgrade" for pycharm editor

        self.num_of_actions = self.envy.num_actions()

        if self.verb>0:
            print(f'\n*** ACTrainer for {self.envy.name} initialized')
            print(f' > actions: {self.envy.num_actions()}, exploration: {self.exploration}')

    # update is performed for both: Actor and Critic
    def update_actor(self, inspect=False):

        batch = self.memory.get_all()
        observations =          self._extract_from_batch(batch, 'observation')
        actions =               self._extract_from_batch(batch, 'action')
        #rewards =               self._extract_from_batch(batch, 'reward')
        dreturns =              self._extract_from_batch(batch, 'dreturn')
        next_observations =     self._extract_from_batch(batch, 'next_observation')
        terminals =             self._extract_from_batch(batch, 'game_over')

        if inspect:
            print(f'\nBatch size: {len(batch)}')
            print(f'observations: {observations.shape}, {observations[0]}')
            print(f'actions: {actions.shape}, {actions[0]}')
            #print(f'rewards {rewards.shape}, {rewards[0]}')
            print(f'next_observations {next_observations.shape}, {next_observations[0]}')
            print(f'terminals {terminals.shape}, {terminals[0]}')

        dreturns_norm = self.zscore_norm(dreturns)

        loss = self.actor.update_batch(
            observations=       observations,
            actions=            actions,
            #rewards=            rewards,
            dreturns=           dreturns_norm,
            next_observations=  next_observations,
            terminals=          terminals,
            discount=           self.discount)

        self.memory.reset()

        return loss