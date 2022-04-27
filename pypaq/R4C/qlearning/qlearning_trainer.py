"""

 2022 (c) piteren

    QLearningTrainer for QLearningActor acting on FiniteActionsRLEnvy

    There are 3 parameters that need to be "constrained":
    batch_size, memory_size and max_ep_steps (for training procedure).
    memory_size should be N*batch size, max_ep_steps probably should be equal to batch_size

"""

import numpy as np
import random
from typing import List

from pypaq.lipytools.plots import two_dim
from pypaq.R4C.envy import FiniteActionsRLEnvy
from pypaq.R4C.trainer import FATrainer, ExperienceMemory
from pypaq.R4C.qlearning.qlearning_actor import QLearningActor


class QLearningTrainer(FATrainer):

    def __init__(
            self,
            actor: QLearningActor,
            envy: FiniteActionsRLEnvy,
            batch_size=         10,
            memsize_batches=    10,
            exploration=        0.5,  # exploration factor
            discount=           0.9,  # discount factor (gamma)
            seed=               123,
            verb=               1):

        self.verb = verb
        random.seed(seed)

        FATrainer.__init__(
            self,
            actor=              actor,
            envy=               envy,
            exploration=        exploration,
            batch_size=         batch_size)
        self.actor = actor # INFO: type "upgrade" for pycharm editor

        self.memsize_batches = memsize_batches
        self.discount = discount

        if self.verb>0: print(f'\n*** QLearningTrainer for {envy.name} (actions: {envy.num_actions()})')

    def init_memory(self):
        self.memory = ExperienceMemory(self.batch_size * self.memsize_batches)

    # updates QLearningActor policy with batch of data from memory
    def update_actor(self):

        batch = self.memory.sample(self.batch_size)

        observations = [me['observation'] for me in batch]
        actions = [me['action'] for me in batch]

        rewards = [me['reward'] for me in batch]
        next_observations = [me['next_observation'] for me in batch]
        no_qvsL = self.actor.get_QVs_batch(next_observations)

        terminals = [me['terminal'] for me in batch]
        no_qvs_terminal = np.zeros(self.envy.num_actions())
        for ix,t in enumerate(terminals):
            if t: no_qvsL[ix] = no_qvs_terminal

        new_qvs = [(r + self.discount * max(no_qvs)) for r,no_qvs in zip(rewards,no_qvsL)]

        self.actor.update_batch(
            observations=   observations,
            actions=        actions,
            new_qvs=        new_qvs)

    def train(
            self,
            num_updates=    2000,   # number of training updates
            test_freq=      100,    # number of updates between test
            test_episodes=  100,    # number of testing episodes
            test_max_steps= 1000,  # max number of episode steps while testing
    ) -> List[float]:

        self.init_memory()

        returnL = []
        for uix in range(num_updates):

            all_rewards = 0
            new_actions = 0
            while new_actions <  self.batch_size:

                observations, actions, rewards = self.play(
                    steps=          self.batch_size - new_actions,
                    break_terminal= True,
                    exploration=    self.exploration,
                    render=         False)

                all_rewards += sum(rewards)

                new_actions += len(observations)
                next_observations = observations[1:] + [self.envy.get_observation()]
                terminals = [False]*(len(observations)-1) + [self.envy.is_terminal()]

                for o,a,r,n,t in zip(observations, actions, rewards, next_observations, terminals):
                    self.memory.append({'observation':o, 'action':a, 'reward':r, 'next_observation':n, 'terminal':t})

            self.update_actor()

            returnL.append(all_rewards)

            if uix % test_freq == 0:
                tr = self.test_on_episodes(
                    n_episodes= test_episodes,
                    max_steps=  test_max_steps)
                if self.verb>0: print(f' > test avg_won:{tr[0]*100:.1f}%, avg_return:{tr[1]:.1f}')

        if self.verb>0: two_dim(returnL)
        return returnL