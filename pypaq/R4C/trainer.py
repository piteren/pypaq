"""

 2022 (c) piteren

    RL Trainer for Actor acting on RLEnvy

"""

from abc import abstractmethod, ABC
from collections import deque
import numpy as np
import random
from typing import List, Tuple, Optional, Dict

from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.plots import two_dim
from pypaq.R4C.helpers import discounted_return, movavg_return
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
            movavg_factor=  0.3,
            verb=           1):

        self.verb = verb
        self.envy = envy
        self.actor = actor
        self.exploration = exploration
        self.batch_size = batch_size
        self.discount = discount
        self.movavg_factor = movavg_factor

        self.memory: Optional[ExperienceMemory] = None
        self.metric = MovAvg()

        if self.verb>0:
            print(f'\n*** Trainer for {self.envy.name} initialized')
            print(f' > exploration: {self.exploration}')
            print(f' > batch_size:  {self.batch_size}')
            print(f' > discount:    {self.discount}')

    def init_memory(self):
        self.memory = ExperienceMemory(self.batch_size)

    # saves sample (given as kwargs - dict) into memory
    def remember(self, **kwargs):
        self.memory.append(kwargs)

    # updates Actor policy, returns Actor "metric" - loss etc. (float)
    def update_actor(self, inspect=False) -> float:
        batch = self.memory.get_all()
        loss = self.actor.update_batch(batch=batch, inspect=inspect)
        self.memory.reset()
        return loss

    # trainer selects exploring action (with Trainer exploratory policy)
    @abstractmethod
    def get_exploring_action(self): pass

    # performs one Actor move (action) and gathers some data
    def _ex_move(
            self,
            exploration=    0.0,    # exploration factor
            sampled=        0.0):   # sampling (vs argmax) factor

        observation = self.envy.get_observation()

        if np.random.rand() < exploration: action = self.get_exploring_action()
        else:                              action = self.actor.get_policy_action(
                                                        observation=    observation,
                                                        #sampled=        sampled)
                                                        sampled=        np.random.rand() < sampled)
        self.envy.run(action)

        reward = self.envy.get_last_action_reward()

        return observation, action, reward

    # plays (envy) until N steps performed, returns (observations, actions, rewards, win/lost)
    def play(
            self,
            steps=          1000,
            break_terminal= False, # for True breaks play at episode end
            exploration=    0.0,
            sampled=        0.0,
            render=         False) -> Tuple[List,List,List]:

        if self.verb>2: print(f'playing for {steps} steps..')
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

        if self.verb>2: print(f'played {len(actions)} steps (break_terminal is {break_terminal})')
        return observations, actions, rewards

    # plays one episode from reset till won or max_steps
    def play_episode(
            self,
            max_steps=      1000,  # single play with max_steps is considered to be won
            exploration=    0.0,
            sampled=        0.0,
            render=         False
    ) -> Tuple[List,List,List,bool]:

        self.envy.reset()
        observations, actions, rewards = self.play(
            steps=          max_steps,
            break_terminal= True,
            exploration=    exploration,
            sampled=        sampled,
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
            upd_on_episode= False,  # updates on episode finish (does not wait till batch)
            train_sampled=  0.3,    # how often move is sampled (vs argmax) while TRAINING
            use_movavg=     True,
            test_freq=      100,    # number of updates between test
            test_episodes=  100,    # number of testing episodes
            test_max_steps= 1000,   # max number of episode steps while testing
            test_render=    True,
            break_ntests=   0,      # when > 0: breaks training after all test episodes succeeded N times
    ) -> Dict:
        """
        generic RL training procedure,
        implementation below is valid for PG & AC algorithm
        may be overridden with custom implementation,
        returns list of Actor "metrics" (Actor loss)
        """

        if self.verb>0: print(f'\nStarting train for {num_updates} updates..')
        self.init_memory()

        returnL = []
        num_act = []
        n_terminals = 0
        last_terminals = 0
        break_succeeded = 0
        for uix in range(num_updates):
            if self.verb in [1,2]: print(f'\rplaying for UPD:{uix:4}', end='')
            if self.verb>2: print(f'playing for UPD:{uix:4}')

            while len(self.memory) < self.batch_size:
                observations, actions, rewards = self.play(
                    steps=          self.batch_size - len(self.memory),
                    break_terminal= True,
                    exploration=    self.exploration,
                    sampled=        train_sampled,
                    render=         False)


                if use_movavg: dreturns = movavg_return(rewards=rewards, factor=self.movavg_factor)
                else:          dreturns = discounted_return(rewards=rewards, discount=self.discount)
                next_observations = observations[1:] + [self.envy.get_observation()]
                terminals = [False]*(len(observations)-1) + [self.envy.is_terminal()]

                # INFO: not all algorithms (PG,AC) need all the data below (we store 'more' just in case)
                for o,a,d,r,n,t in zip(observations, actions, dreturns, rewards, next_observations, terminals):
                    self.remember(observation=o, action=a, dreturn=d, reward=r, next_observation=n, game_over=t)

                if self.verb>2: print(f' >> Trainer gots {len(observations):3} observations after play and {len(self.memory):3} in memory' )

                num_act.append(len(actions))
                if self.envy.is_terminal(): n_terminals += 1
                if upd_on_episode: break

            inspect = (uix % test_freq == 0) if self.verb>1 else False
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

                if tr[0] == 1: break_succeeded += 1
                else: break_succeeded = 0

            if break_ntests and break_succeeded==break_ntests: break

        if self.verb>1:
            two_dim(returnL, name='Actor loss')
            two_dim(num_act, name='num_act')

        return { # training_report
            'returnL':          returnL,
            'n_terminals':      n_terminals,
            'break_succeeded':  break_succeeded}


# RL Trainer for Actor acting on FiniteActionsRLEnvy
class FATrainer(Trainer, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            **kwargs):
        Trainer.__init__(self, envy=envy, **kwargs)
        self.envy = envy # INFO: type "upgrade" for pycharm editor
        self.num_of_actions = self.envy.num_actions()

    # selects random action from action space
    def get_exploring_action(self):
        return np.random.choice(self.num_of_actions)