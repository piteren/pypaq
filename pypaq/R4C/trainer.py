"""

 2022 (c) piteren

    RL Trainer for Actor acting on RLEnvy

"""

from abc import abstractmethod, ABC
from collections import deque
import numpy as np
import random
from typing import List, Tuple, Optional

from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.plots import two_dim
from pypaq.R4C.envy import RLEnvy, FiniteActionsRLEnvy
from pypaq.R4C.actor import Actor


# Trainer Experience Memory
class ExperienceMemory:

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.memory = deque(maxlen=self.maxsize)

    def reset(self):
        self.memory = deque(maxlen=self.maxsize)

    def append(self, element:dict):
        self.memory.append(element)

    def sample(self, n) -> List[dict]:
        return random.sample(self.memory, n)

    def get_all(self) -> List[dict]:
        return list(self.memory)

    def __len__(self):
        return len(self.memory)


# RL Trainer for Actor acting on RLEnvy
class Trainer(ABC):

    def __init__(
            self,
            envy: RLEnvy,
            actor: Actor,
            exploration=    0.5,    # exploration factor
            batch_size=     32,     # Actor update data size
            discount=       0.9,    # return discount factor (gamma)
            verb=           1):

        self.verb = verb
        self.envy = envy
        self.actor = actor
        self.exploration = exploration
        self.batch_size = batch_size
        self.discount = discount

        self.memory: Optional[ExperienceMemory] = None
        self.metric = MovAvg()

        if self.verb>0: print(f'\n*** Trainer for {self.envy.name} initialized')

    def init_memory(self):
        self.memory = ExperienceMemory(self.batch_size)

    # saves sample (given as kwargs - dict) into memory
    def remember(self, **kwargs):
        self.memory.append(kwargs)

    # extracts array of data from a batch
    @staticmethod
    def _extract_from_batch(
            batch :List[dict],
            key: str) -> np.ndarray:
        return np.array(list(map(lambda x: x[key], batch)))

    # prepares list of discounted accumulated return from [reward]
    def discounted_return(self, rewards: List[float]) -> List[float]:

        dar = np.zeros_like(rewards)
        s = 0.0
        for i in reversed(range(len(rewards))):
            s = s * self.discount + rewards[i]
            dar[i] = s

        return list(dar)

    # normalizes x with zscore (0 mean 1 std), this is helpful for training, as rewards can vary considerably between episodes,
    @staticmethod
    def zscore_norm(x):
        if len(x) < 2: return x
        return (x - np.mean(x)) / np.std(x) + 0.00000001

    # updates Actor policy
    @abstractmethod
    def update_actor(self, inspect=False): pass

    # trainer selects exploring action (with Trainer exploratory policy)
    @abstractmethod
    def get_exploring_action(self): pass

    # performs one Actor move (action) and gathers some data
    def _ex_move(
            self,
            exploration=    0.0,
            sampled=        False):

        observation = self.envy.get_observation()

        if np.random.rand() < exploration: action = self.get_exploring_action()
        else:                              action = self.actor.get_policy_action(
                                                        observation=    observation,
                                                        sampled=        sampled)
        self.envy.run(action)

        reward = self.envy.get_last_action_reward()

        return observation, action, reward

    # plays (envy) until N steps performed, returns (observations, actions, rewards, win/lost)
    def play(
            self,
            steps=          1000,
            break_terminal= False, # for True breaks play at episode end
            exploration=    0.0,
            sampled=        False,
            render=         False) -> Tuple[List,List,List]:

        observations = []
        actions = []
        rewards = []

        if self.envy.is_terminal(): self.envy.reset()

        while len(actions) < steps:

            if self.envy.is_terminal():
                if break_terminal: break
                self.envy.reset()

            observation, action, reward = self._ex_move(
                exploration=    exploration,
                sampled=        sampled)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)

            if render: self.envy.render()

        return observations, actions, rewards

    # plays one episode from reset till won or max_steps, returns (observations, actions, rewards, win/lost)
    def play_episode(
            self,
            max_steps=      1000,  # single play with max_steps is considered to be won
            exploration=    0.0,
            render=         False
    ) -> Tuple[List,List,List,bool]:

        self.envy.reset()
        observations, actions, rewards = self.play(
            steps=          max_steps,
            break_terminal= True,
            exploration=    exploration,
            render=         render)

        won = self.envy.won_episode() or len(actions)==max_steps

        return observations, actions, rewards, won

    # plays n episodes, returns (won_factor, avg/reward)
    def test_on_episodes(
            self,
            n_episodes= 100,
            max_steps=  1000
    ) -> Tuple[float,float]:
        won = 0
        reward = 0
        for e in range(n_episodes):
            epd = self.play_episode(
                max_steps=      max_steps,
                exploration=    0.0)
            won += int(epd[3])
            reward += sum(epd[2])
        return won/n_episodes, reward/n_episodes

    # generic RL training procedure
    def train(
            self,
            num_updates=    2000,   # number of training updates
            upd_on_episode= False,  # updates on episode finish (do not wait till batch)
            test_freq=      100,    # number of updates between test
            test_episodes=  100,    # number of testing episodes
            test_max_steps= 1000,   # max number of episode steps while testing
            test_render=    True) -> List[float]:
        """
        generic RL training procedure,
        implementation below is valid for PG & AC algorithm
        usually to be overridden with custom implementation,
        returns list of "metrics" (Actor loss)
        """

        print(f'\nStarting train for {num_updates} updates..')
        self.init_memory()

        returnL = []
        n_terminals = 0
        last_terminals = 0
        for uix in range(num_updates):
            print(f'\rUPD:{uix:4}', end='')

            while len(self.memory) < self.batch_size:
                observations, actions, rewards = self.play(
                    steps=          self.batch_size - len(self.memory),
                    break_terminal= True,
                    exploration=    self.exploration,
                    sampled=        True,
                    render=         False)

                dreturns = self.discounted_return(rewards)
                next_observations = observations[1:] + [self.envy.get_observation()]
                terminals = [False]*(len(observations)-1) + [self.envy.is_terminal()]

                # INFO: not all algorithms (PG,AC) need all below data, but to make this method proper for more algorithms we store 'more'
                for o,a,d,r,n,t in zip(observations, actions, dreturns, rewards, next_observations, terminals):
                    self.remember(observation=o, action=a, dreturn=d, reward=r, next_observation=n, game_over=t)

                n_terminals += np.sum(np.array(terminals).astype(dtype=np.int32))

                if upd_on_episode: break

            inspect = (uix % test_freq == 0) if self.verb > 1 else False
            loss_actor = self.update_actor(inspect=inspect)
            returnL.append(self.metric.upd(loss_actor))

            if uix % test_freq == 0:

                observations, actions, rewards, won = self.play_episode(
                    max_steps=      test_max_steps,
                    exploration=    0.0,
                    render=         test_render)

                tr = self.test_on_episodes(
                    n_episodes= test_episodes,
                    max_steps=  test_max_steps)

                if self.verb>0: print(f' T:{n_terminals}(+{n_terminals-last_terminals}) -- TS: {len(actions)} actions, return {sum(rewards):.1f} ({"won" if won else "lost"}) -- {test_episodes}xTS: avg_won: {tr[0]*100:.1f}%, avg_return: {tr[1]:.1f} -- loss_actor: {self.metric():.4f}')
                last_terminals = n_terminals

        if self.verb>0: two_dim(returnL, name='Actor loss')
        return returnL


# RL Trainer for Actor acting on FiniteActionsRLEnvy
class FATrainer(Trainer, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            **kwargs):
        Trainer.__init__(self, envy=envy, **kwargs)
        self.envy = envy # INFO: type "upgrade" for pycharm editor

    def get_exploring_action(self):
        return np.random.choice(self.envy.num_actions())