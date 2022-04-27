import numpy as np

from pypaq.R4C.policy_gradients.pg_actor import PGActor
from pypaq.R4C.policy_gradients.actor_critic.ac_critic import ACCritic
from pypaq.R4C.trainer import FATrainer


class ACTrainer(FATrainer):

    def __init__(
            self,
            actor: PGActor,
            critic_class: type(ACCritic),
            critic_mdict: dict,
            verb=           1,
            **kwargs):

        FATrainer.__init__(self, actor=actor, verb=verb, **kwargs)
        self.actor = actor # INFO: type "upgrade" for pycharm editor

        self.critic = critic_class(
            num_actions=    self.envy.num_actions(),
            observation=    self.envy.get_observation(),
            mdict=          critic_mdict,
            verb=           self.verb)

        self.num_of_actions = self.envy.num_actions()

        if self.verb>0:
            print(f'\n*** ACTrainer for {self.envy.name} initialized')
            print(f' > actions: {self.envy.num_actions()}, exploration: {self.exploration}')

    # converts one dim arr of ints into two dim one-hot array
    def _actions_OH_encoding(self, actions:np.array) -> np.ndarray:
        hot = np.zeros((len(actions), self.num_of_actions))
        hot[np.arange(len(actions)), actions] = 1
        return hot

    # sets terminal states QVs to zeroes
    def _update_terminal_QVs(self, qvs, terminals):
        zeroes = np.zeros(self.num_of_actions)
        for i in range(len(terminals)):
            if terminals[i]:
                qvs[i] = zeroes
        return qvs

    # update is performed for both: Actor and Critic
    def update_actor(self, inspect=False):

        batch = self.memory.get_all()
        observations =          self._extract_from_batch(batch, 'observation')
        actions =               self._extract_from_batch(batch, 'action')
        rewards =               self._extract_from_batch(batch, 'reward')
        next_observations =     self._extract_from_batch(batch, 'next_observation')
        terminals =             self._extract_from_batch(batch, 'game_over')

        if inspect:
            print(f'\nBatch size: {len(batch)}')
            print(f'observations: {observations.shape}, {observations[0]}')
            print(f'actions: {actions.shape}, {actions[0]}')
            print(f'rewards {rewards.shape}, {rewards[0]}')
            print(f'next_observations {next_observations.shape}, {next_observations[0]}')
            print(f'terminals {terminals.shape}, {terminals[0]}')

        # get next_observations actions_probs (with Actor policy)
        next_actions_probs = self.actor.get_policy_probs_batch(next_observations)
        if inspect: print(f'next_actions_probs {next_actions_probs.shape}, {next_actions_probs[0]}')

        # get QVs of current observations
        qvss = self.critic.get_qvs_batch(observations)
        qv_actions = qvss[np.arange(actions.shape[-1]),actions] # get QV of selected actions
        if inspect:
            print(f'qvss {qvss.shape}, {qvss[0]}')
            print(f'qv_actions {qv_actions.shape}, {qv_actions[0]}')

        # update Actor
        loss_actor = self.actor.update_batch(
            observations=   observations,
            actions=        actions,
            dreturns=       qv_actions)

        actions_OH = self._actions_OH_encoding(actions)
        if inspect: print(f'actions_OH {actions_OH.shape}, {actions_OH[0]}')

        # get QVs of next observations
        next_actions_qvs = self.critic.get_qvs_batch(next_observations)
        next_actions_qvs = self._update_terminal_QVs(next_actions_qvs, terminals=terminals)
        if inspect: print(f'next_action_qvs {next_actions_qvs.shape}, {next_actions_qvs[0]}')

        # update Critic
        loss_critic = self.critic.update_batch(
            observations=       observations,
            actions_OH=         actions_OH,
            next_action_qvs=    next_actions_qvs,
            next_actions_probs= next_actions_probs,
            rewards=            rewards)

        if np.isnan(loss_actor) or np.isnan(loss_critic): raise Exception('NaN cost!')

        self.memory.reset()

        return loss_actor