"""

 2022 (c) piteren

    RL Trainer for Actor acting on RLEnvy

        Implements generic RL training procedure, that is valid for some algorithms (QTable, PG, AC).
        May be overridden with custom implementation, returns Dict with some training stats

"""

from abc import abstractmethod, ABC
from collections import deque
import numpy as np
import random
import time
from typing import List, Tuple, Optional

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.moving_average import MovAvg
from pypaq.R4C.envy import RLEnvy, FiniteActionsRLEnvy
from pypaq.R4C.actor import TrainableActor
from pypaq.R4C.helpers import RLException
from pypaq.torchness.tbwr import TBwr
from pypaq.torchness.comoneural.avg_probs import avg_probs
from pypaq.torchness.comoneural.zeroes_processor import ZeroesProcessor


# Trainer Experience Memory (deque od dicts)
class ExperienceMemory:

    def __init__(
            self,
            maxsize: int,
            seed: int):
        self.memory = deque(maxlen=maxsize)
        random.seed(seed)

    def append(self, element:dict):
        self.memory.append(element)

    # returns random sample of non-duplicates from memory
    def sample(self, n) -> List[dict]:
        return random.sample(self.memory, n)

    # returns all elements from memory in original order
    def get_all(self) -> List[dict]:
        return list(self.memory)

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


# Reinforcement Learning Trainer for Actor acting on RLEnvy
class RLTrainer(ABC):

    def __init__(
            self,
            envy: RLEnvy,
            actor: TrainableActor,
            batch_size: int,        # Actor update batch data size
            memsize_batches: int,   # ExperienceMemory size (in number of batches)
            exploration: float,     # train exploration factor
            train_sampled: float,   # how often move is sampled (vs argmax) while training
            seed: int=      123,
            logger=         None,
            loglevel=       20,
            hpmser_mode=    False):

        self._rlog = logger or get_pylogger(level=loglevel)
        self._rlog.info(f'*** RLTrainer *** initializes..')
        self._rlog.info(f'> Envy:          {envy.__class__.__name__}')
        self._rlog.info(f'> Actor:         {actor.__class__.__name__}, name: {actor.name}')
        self._rlog.info(f'> batch_size:    {batch_size}')
        self._rlog.info(f'> memory size:   {batch_size*memsize_batches}')
        self._rlog.info(f'> exploration:   {exploration}')
        self._rlog.info(f'> train_sampled: {train_sampled}')
        self._rlog.info(f'> seed:          {seed}')

        self.envy = envy
        self.actor = actor
        self.batch_size = batch_size
        self.memsize = self.batch_size * memsize_batches
        self.exploration = exploration
        self.train_sampled = train_sampled
        self.memory: Optional[ExperienceMemory] = None
        self.seed = seed
        np.random.seed(self.seed)
        self.hpmser_mode = hpmser_mode

        self._tbwr = TBwr(logdir=self.actor.get_save_dir()) if not self.hpmser_mode else None
        self._upd_step = 0 # global Trainer update step

        self._zepro = ZeroesProcessor(
            intervals=  (10,50,100),
            tbwr=       self._tbwr) if not self.hpmser_mode else None

    # updates Actor policy, returns dict with Actor "metrics" - loss etc.
    def _update_actor(self, inspect=False) -> dict:
        batch = self.memory.sample(self.batch_size)
        return self.actor.update_with_experience(
            batch=      batch,
            inspect=    inspect)

    # trainer selects exploring action (with Trainer exploratory policy)
    @abstractmethod
    def _get_exploring_action(self) -> object: pass

    # performs one Actor move (action)
    def _ex_move(
            self,
            exploration=    0.0,    # prob pf exploration
            sampled=        0.0     # prob of sampling (vs argmax)
    ) -> Tuple[object,object,float,bool,bool]:

        pre_action_observation = self.envy.get_observation()

        if np.random.rand() < exploration: action = self._get_exploring_action()
        else:                              action = self.actor.get_policy_action(
                                                        observation=    pre_action_observation,
                                                        sampled=        np.random.rand() < sampled)
        reward, is_terminal, won = self.envy.run(action)

        return pre_action_observation, action, reward, is_terminal, won

    # plays (envy) until N steps performed or terminal state
    def play(
            self,
            reset: bool,            # for True starts play from the initial state
            steps: int,
            break_terminal: bool,   # for True breaks play at terminal state
            exploration: float,
            sampled: float,
            render: bool) -> Tuple[List[object], List[object], List[float], List[bool], List[bool]]:

        self._rlog.log(5,f'playing for {steps} steps..')

        if reset: self.envy.reset()

        observations = []
        actions = []
        rewards = []
        terminals = []
        wons = []

        while len(actions) < steps:

            observation, action, reward, is_terminal, won = self._ex_move(
                exploration=    exploration,
                sampled=        sampled)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(is_terminal)
            wons.append(won)

            if render: self.envy.render()

            if is_terminal:
                if break_terminal: break
                self.envy.reset()

        self._rlog.log(5,f'played {len(actions)} steps (break_terminal is {break_terminal})')
        return observations, actions, rewards, terminals, wons

    # plays one episode from reset till terminal state or max_steps
    def _play_episode(
            self,
            exploration: float,
            sampled: float,
            render: bool,
            max_steps: Optional[int]=   None,  # if max steps is given then single play for max_steps is considered to be won
    ) -> Tuple[List[object], List[object], List[float], bool]:

        if max_steps is None and self.envy.get_max_steps() is None:
            raise RLException('Cannot play episode for Envy where max_steps is None and given max_steps is also None')

        observations, actions, rewards, terminals, wons = self.play(
            steps=          max_steps or self.envy.get_max_steps(),
            reset=          True,
            break_terminal= True,
            exploration=    exploration,
            sampled=        sampled,
            render=         render)

        return observations, actions, rewards, wons[-1]

    # plays n episodes, returns (won_factor, avg/reward)
    def test_on_episodes(
            self,
            n_episodes=                 100,
            max_steps: Optional[int]=   None,
    ) -> Tuple[float, float]:
        n_won = 0
        sum_rewards = 0
        for e in range(n_episodes):
            observations, actions, rewards, won = self._play_episode(
                exploration=    0.0,
                sampled=        0.0,
                render=         False,
                max_steps=      max_steps)
            n_won += int(won)
            sum_rewards += sum(rewards)
        return n_won/n_episodes, sum_rewards/n_episodes

    # generic RL training procedure
    def train(
            self,
            num_updates: int,                       # number of training updates
            upd_on_episode=                 False,  # updates on episode finish / terminal (does not wait till batch)
            test_freq=                      100,    # number of updates between test
            test_episodes: int=             100,    # number of testing episodes
            test_max_steps: Optional[int]=  None,   # max number of episode steps while testing
            test_render: bool=              False,  # renders one episode while test
            inspect: bool=                  False,  # for debug / research
            break_ntests: Optional[int]=    None,   # breaks training after all test episodes succeeded N times in a row
    ) -> dict:

        stime = time.time()
        self._rlog.info(f'Starting train for {num_updates} updates..')

        self.memory = ExperienceMemory(
            maxsize=    self.memsize,
            seed=       self.seed)
        self._rlog.info(f'> initialized ExperienceMemory of maxsize {self.memsize}')

        self.envy.reset()
        loss_mavg = MovAvg()
        lossL = []
        n_actions = 0               # total number of train actions
        n_terminals = 0             # number of terminal states reached while training
        last_terminals = 0          # previous number of terminal states
        n_won = 0                   # number of wins while training
        succeeded_row_curr = 0      # current number of succeeded tests in a row
        succeeded_row_max = 0       # max number of succeeded tests in a row
        for uix in range(num_updates):

            # get a batch of data
            new_actions = 0
            while new_actions < self.batch_size:

                observations, actions, rewards, terminals, wons = self.play(
                    steps=          self.batch_size - new_actions,
                    reset=          False,
                    break_terminal= True,
                    exploration=    self.exploration,
                    sampled=        self.train_sampled,
                    render=         False)

                new_actions += len(actions)
                n_actions += len(actions)

                next_observations = observations[1:] + [self.envy.get_observation()]

                # INFO: not all algorithms (QLearning,PG,AC) need all the data below (we store 'more' just in case)
                for o,a,r,n,t in zip(observations, actions, rewards, next_observations, terminals):
                    self.memory.append(dict(observation=o, action=a, reward=r, next_observation=n, terminal=t))

                if terminals[-1]: n_terminals += 1 # ..may not be terminal when limit of new_actions reached
                if wons[-1]: n_won += 1

                self._rlog.debug(f' >> Trainer gots {len(observations):3} observations after play and {len(self.memory):3} in memory, new_actions: {new_actions}' )

                if upd_on_episode: break

            # update Actor & process metrics
            upd_metrics = self._update_actor(inspect=inspect and uix % test_freq == 0)
            self._upd_step += 1

            if 'loss' in upd_metrics: lossL.append(loss_mavg.upd(upd_metrics['loss']))

            # process / monitor policy probs
            if self._tbwr and 'probs' in upd_metrics:
                for k,v in avg_probs(upd_metrics.pop('probs')).items():
                    self._tbwr.add(value=v, tag=f'actor_upd/{k}', step=self._upd_step)

            if self._zepro and 'zeroes' in upd_metrics:
                self._zepro.process(zs=upd_metrics.pop('zeroes'))

            if self._tbwr:
                for k,v in upd_metrics.items():
                    self._tbwr.add(value=v, tag=f'actor_upd/{k}', step=self._upd_step)

            # test Actor
            if uix % test_freq == 0:

                # single episode
                observations, actions, rewards, won = self._play_episode(
                    exploration=    0.0,
                    sampled=        0.0,
                    render=         test_render,
                    max_steps=      test_max_steps)

                # few tests
                avg_won, avg_return = self.test_on_episodes(
                    n_episodes=     test_episodes,
                    max_steps=      test_max_steps)

                self._rlog.info(f'# {uix:3} term:{n_terminals}(+{n_terminals-last_terminals}) -- TS: {len(actions)} actions, return {sum(rewards):.1f} ({"won" if won else "lost"}) -- {test_episodes}xTS: avg_won: {avg_won*100:.1f}%, avg_return: {avg_return:.1f} -- loss_actor: {loss_mavg():.4f}')
                last_terminals = n_terminals

                if avg_won == 1:
                    succeeded_row_curr += 1
                    if succeeded_row_curr > succeeded_row_max: succeeded_row_max = succeeded_row_curr
                else: succeeded_row_curr = 0

            if break_ntests is not None and succeeded_row_curr==break_ntests: break

        self._rlog.info(f'### Training finished, time taken: {time.time()-stime:.2f}sec')

        return { # training_report
            'n_actions':            n_actions,
            'lossL':                lossL,
            'n_terminals':          n_terminals,
            'n_won':                n_won,
            'succeeded_row_max':    succeeded_row_max}


# FiniteActions RL Trainer (for Actor acting on FiniteActionsRLEnvy)
class FATrainer(RLTrainer, ABC):

    def __init__(self, envy:FiniteActionsRLEnvy, seed:int, **kwargs):
        np.random.seed(seed)
        RLTrainer.__init__(self, envy=envy, seed=seed, **kwargs)
        self.envy = envy  # INFO: type "upgrade" for pycharm editor

        self._rlog.info(f'*** FATrainer *** initialized')
        self._rlog.info(f'> number of actions: {self.envy.num_actions()}')

    # selects 100% random action from action space, (np. seed is fixed at Trainer)
    def _get_exploring_action(self):
        return np.random.choice(self.envy.num_actions())