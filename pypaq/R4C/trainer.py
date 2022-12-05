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
from typing import List, Tuple, Optional, Dict

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.moving_average import MovAvg
from pypaq.R4C.envy import RLEnvy, FiniteActionsRLEnvy
from pypaq.R4C.actor import TrainableActor
from pypaq.torchness.base_elements import TBwr
from pypaq.comoneural.zeroes_processor import ZeroesProcessor


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
            seed: int=  123,
            logger=     None,
            loglevel=   20):

        self._log = logger or get_pylogger(level=loglevel)
        self._log.info(f'*** RLTrainer *** initializes..')
        self._log.info(f'> Envy:          {envy.__class__.__name__}')
        self._log.info(f'> Actor:         {actor.__class__.__name__}, name: {actor.name}')
        self._log.info(f'> batch_size:    {batch_size}')
        self._log.info(f'> memory size:   {batch_size*memsize_batches}')
        self._log.info(f'> exploration:   {exploration}')
        self._log.info(f'> train_sampled: {train_sampled}')
        self._log.info(f'> seed:          {seed}')

        self.envy = envy
        self.actor = actor
        self.batch_size = batch_size
        self.memsize = self.batch_size * memsize_batches
        self.exploration = exploration
        self.train_sampled = train_sampled
        self.memory: Optional[ExperienceMemory] = None
        self.seed = seed
        np.random.seed(self.seed)

        self._TBwr = TBwr(logdir=self.actor.get_save_dir()) # TensorBoard writer
        self._upd_step = 0                                  # global update step

        self.zepro = ZeroesProcessor(
            intervals=  (10,50,100),
            tbwr=       self._TBwr)

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
    ) -> Tuple[object, object, float]:

        observation = self.envy.get_observation()

        if np.random.rand() < exploration: action = self._get_exploring_action()
        else:                              action = self.actor.get_policy_action(
                                                        observation=    observation,
                                                        sampled=        np.random.rand() < sampled)
        self.envy.run(action)
        reward = self.envy.get_last_action_reward()

        return observation, action, reward

    # plays (envy) until N steps performed, returns (observations, actions, rewards)
    def play(
            self,
            steps=          1000,
            break_terminal= False, # for True breaks play at episode end
            exploration=    0.0,
            sampled=        0.0,
            render=         False) -> Tuple[List[object], List[object], List[float]]:

        self._log.log(5,f'playing for {steps} steps..')
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

        self._log.log(5,f'played {len(actions)} steps (break_terminal is {break_terminal})')
        return observations, actions, rewards

    # plays one episode from reset till won or max_steps
    def play_episode(
            self,
            max_steps=      1000,  # single play with max_steps is considered to be won
            exploration=    0.0,
            sampled=        0.0,
            render=         False
    ) -> Tuple[List[object], List[object], List[float], bool]:

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
    ) -> Tuple[float, float]:
        won = 0
        reward = 0
        for e in range(n_episodes):
            epd = self.play_episode(
                max_steps=      max_steps,
                exploration=    0.0)
            won += int(epd[3])
            reward += sum(epd[2])
        self._log.debug(f'Test on {n_episodes} episodes with {max_steps} max_steps finished: WON: {int(won/n_episodes*100)}% avg reward: {reward/n_episodes:.3f}')
        return won/n_episodes, reward/n_episodes

    # generic RL training procedure
    def train(
            self,
            num_updates: int,           # number of training updates
            upd_on_episode=     False,  # updates on episode finish / terminal (does not wait till batch)
            test_freq=          100,    # number of updates between test
            test_episodes=      100,    # number of testing episodes
            test_max_steps=     1000,   # max number of episode steps while testing
            test_render=        False,  # renders one episode while test
            inspect=            False,  # for debug / research
            break_ntests=       0,      # when > 0: breaks training after all test episodes succeeded N times in a row
    ) -> dict:

        stime = time.time()
        self._log.info(f'Starting train for {num_updates} updates..')

        self.memory = ExperienceMemory(
            maxsize=    self.memsize,
            seed=       self.seed)
        self._log.info(f'> initialized ExperienceMemory of maxsize {self.memsize}')

        loss_mavg = MovAvg()
        lossL = []
        n_terminals = 0             # number of terminal states reached while training
        last_terminals = 0          # previous number of terminal states
        n_won = 0                   # number of wins while training
        succeeded_row_curr = 0      # current number of succeeded tests in a row
        succeeded_row_max = 0       # max number of succeeded tests in a row
        for uix in range(num_updates):

            # get a batch of data
            new_actions = 0
            while new_actions < self.batch_size:

                observations, actions, rewards = self.play(
                    steps=          self.batch_size - new_actions,
                    break_terminal= True,
                    exploration=    self.exploration,
                    sampled=        self.train_sampled,
                    render=         False)

                is_terminal = self.envy.is_terminal()  # may not be when limit of new_actions reached
                new_actions += len(actions)

                next_observations = observations[1:] + [self.envy.get_observation()]
                terminals = [False]*(len(observations)-1) + [is_terminal]

                # INFO: not all algorithms (QLearning,PG,AC) need all the data below (we store 'more' just in case)
                for o,a,r,n,t in zip(observations, actions, rewards, next_observations, terminals):
                    self.memory.append(dict(observation=o, action=a, reward=r, next_observation=n, terminal=t))

                self._log.debug(f' >> Trainer gots {len(observations):3} observations after play and {len(self.memory):3} in memory, new_actions: {new_actions}' )

                if is_terminal: n_terminals += 1
                if self.envy.won_episode(): n_won += 1

                if upd_on_episode: break

            # update Actor
            upd_metrics = self._update_actor(inspect=inspect and uix % test_freq == 0)
            self._upd_step += 1
            if 'loss' in upd_metrics: lossL.append(loss_mavg.upd(upd_metrics['loss']))

            if 'zeroes' in upd_metrics:
                self.zepro.process(zs=upd_metrics.pop('zeroes'))

            for k in upd_metrics:
                self._TBwr.add(
                    value=  upd_metrics[k],
                    tag=    f'actor_upd/{k}',
                    step=   self._upd_step)

            # test Actor
            if uix % test_freq == 0:

                observations, actions, rewards, won = self.play_episode(
                    max_steps=      test_max_steps,
                    exploration=    0.0,
                    render=         test_render)

                ts_res = self.test_on_episodes(
                    n_episodes= test_episodes,
                    max_steps=  test_max_steps)

                self._log.info(f'# {uix:3} term:{n_terminals}(+{n_terminals-last_terminals}) -- TS: {len(actions)} actions, return {sum(rewards):.1f} ({"won" if won else "lost"}) -- {test_episodes}xTS: avg_won: {ts_res[0]*100:.1f}%, avg_return: {ts_res[1]:.1f} -- loss_actor: {loss_mavg():.4f}')
                last_terminals = n_terminals

                if ts_res[0] == 1:
                    succeeded_row_curr += 1
                    if succeeded_row_curr > succeeded_row_max: succeeded_row_max = succeeded_row_curr
                else: succeeded_row_curr = 0

            if break_ntests and succeeded_row_curr==break_ntests: break

        self._log.info(f'### Training finished, time taken: {time.time()-stime:.2f}sec')

        return { # training_report
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

        self._log.info(f'*** FATrainer *** initialized')
        self._log.info(f'> number of actions: {self.envy.num_actions()}')

    # selects 100% random action from action space, (np. seed is fixed at Trainer)
    def _get_exploring_action(self):
        return np.random.choice(self.envy.num_actions())