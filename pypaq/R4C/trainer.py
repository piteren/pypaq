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

from pypaq.lipytools.moving_average import MovAvg
from pypaq.lipytools.plots import two_dim
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.R4C.helpers import discounted_return, movavg_return
from pypaq.R4C.envy import RLEnvy, FiniteActionsRLEnvy
from pypaq.R4C.actor import TrainableActor


# Trainer Experience Memory (deque od dicts)
class ExperienceMemory:

    def __init__(self, maxsize):
        self.memory = deque(maxlen=maxsize)

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


# RL Trainer for Actor acting on RLEnvy
class Trainer(ABC):

    def __init__(
            self,
            envy: RLEnvy,
            actor: TrainableActor,
            logger=     None):

        if not logger: logger = get_pylogger(name='Trainer')
        self.__log = logger

        self.envy = envy
        self.actor = actor
        self.memory: Optional[ExperienceMemory] = None

        self.__log.info(f'*** Trainer for {self.envy.name} initialized')

    # updates Actor policy, returns Actor "metric" - loss etc. (float), baseline implementation
    def update_actor(
            self,
            batch_size: int,
            discount: float,
            inspect=    False) -> float:
        batch = self.memory.sample(batch_size)
        return self.actor.update_with_experience(
            batch=      batch,
            inspect=    inspect)

    # trainer selects exploring action (with Trainer exploratory policy)
    @abstractmethod
    def get_exploring_action(self): pass

    # performs one Actor move (action)
    def _ex_move(
            self,
            exploration=    0.0,    # prob pf exploration
            sampled=        0.0     # prob of sampling (vs argmax)
    ) -> Tuple[object, object, float]:

        observation = self.envy.get_observation()

        if np.random.rand() < exploration: action = self.get_exploring_action()
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

        self.__log.log(5,f'playing for {steps} steps..')
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

        self.__log.log(5,f'played {len(actions)} steps (break_terminal is {break_terminal})')
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
        self.__log.debug(f'Test on {n_episodes} episodes with {max_steps} max_steps finished: WON: {int(won/n_episodes*100)}% avg reward: {reward/n_episodes:.3f}')
        return won/n_episodes, reward/n_episodes

    # generic RL training procedure
    def train(
            self,
            num_updates=        2000,   # number of training updates
            batch_size=         32,     # Actor update data size
            upd_on_episode=     False,  # updates on episode finish / terminal (does not wait till batch)
            memsize_batches=    1,      # ExperienceMemory size (in number of batches)
            exploration=        0.5,    # exploration factor
            discount=           0.9,    # return discount factor (gamma)
            train_sampled=      0.3,    # how often move is sampled (vs argmax) while training
            movavg_factor=      0.3,
            use_movavg=         True,
            test_freq=          100,    # number of updates between test
            test_episodes=      100,    # number of testing episodes
            test_max_steps=     1000,   # max number of episode steps while testing
            test_render=        False,
            break_ntests=       0,      # when > 0: breaks training after all test episodes succeeded N times in a row
    ) -> Dict:

        stime = time.time()
        self.__log.info(f'Starting train for {num_updates} updates..')
        self.__log.info(f'> batch_size:    {batch_size}')
        self.__log.info(f'> exploration:   {exploration}')
        self.__log.info(f'> discount:      {discount}')
        self.__log.info(f'> train_sampled: {train_sampled}')
        self.__log.info(f'> movavg_factor: {movavg_factor}, used: {use_movavg}')

        mem_size = memsize_batches * batch_size
        self.memory = ExperienceMemory(mem_size)
        self.__log.info(f'> initialized ExperienceMemory of maxsize {mem_size}')

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
            while new_actions < batch_size:

                observations, actions, rewards = self.play(
                    steps=          batch_size - new_actions,
                    break_terminal= True,
                    exploration=    exploration,
                    sampled=        train_sampled,
                    render=         False)

                new_actions += len(observations)

                # TODO: move to prepare_batch_from_experience()
                if use_movavg: dreturns = movavg_return(rewards=rewards, factor=movavg_factor)
                else:          dreturns = discounted_return(rewards=rewards, discount=discount)

                next_observations = observations[1:] + [self.envy.get_observation()]
                terminals = [False]*(len(observations)-1) + [self.envy.is_terminal()]

                # INFO: not all algorithms (QLearning,PG,AC) need all the data below (we store 'more' just in case)
                for o,a,d,r,n,t in zip(observations, actions, dreturns, rewards, next_observations, terminals):
                    self.memory.append(dict(observation=o, action=a, dreturn=d, reward=r, next_observation=n, terminal=t))

                self.__log.debug(f' >> Trainer gots {len(observations):3} observations after play and {len(self.memory):3} in memory, new_actions: {new_actions}' )

                if self.envy.is_terminal(): n_terminals += 1
                if self.envy.won_episode(): n_won += 1

                if upd_on_episode: break

            # update Actor
            # batch_from_experience = self.prepare_batch_from_experience(batch_size=batch_size)
            loss_actor = self.update_actor(
                batch_size= batch_size,
                discount=   discount,
                inspect=    (uix % test_freq == 0) if self.__log.getEffectiveLevel()<20 else False)
            lossL.append(loss_mavg.upd(loss_actor))

            # test Actor
            if uix % test_freq == 0:

                observations, actions, rewards, won = self.play_episode(
                    max_steps=      test_max_steps,
                    exploration=    0.0,
                    render=         test_render)

                ts_res = self.test_on_episodes(
                    n_episodes= test_episodes,
                    max_steps=  test_max_steps)

                self.__log.info(f' term:{n_terminals}(+{n_terminals-last_terminals}) -- TS: {len(actions)} actions, return {sum(rewards):.1f} ({"won" if won else "lost"}) -- {test_episodes}xTS: avg_won: {ts_res[0]*100:.1f}%, avg_return: {ts_res[1]:.1f} -- loss_actor: {loss_mavg():.4f}')
                last_terminals = n_terminals

                if ts_res[0] == 1:
                    succeeded_row_curr += 1
                    if succeeded_row_curr > succeeded_row_max: succeeded_row_max = succeeded_row_curr
                else: succeeded_row_curr = 0

            if break_ntests and succeeded_row_curr==break_ntests: break

        self.__log.info(f'Training finished, time taken: {time.time()-stime:.2f}sec')
        if self.__log.getEffectiveLevel()<30: two_dim(lossL, name='Actor loss')

        return { # training_report
            'lossL':                lossL,
            'n_terminals':          n_terminals,
            'n_won':                n_won,
            'succeeded_row_max':    succeeded_row_max}


# FiniteActions RL Trainer (for Actor acting on FiniteActionsRLEnvy)
class FATrainer(Trainer, ABC):

    def __init__(
            self,
            envy: FiniteActionsRLEnvy,
            logger= None,
            **kwargs):

        if not logger: logger = get_pylogger(name='FATrainer')
        self.__log = logger

        Trainer.__init__(
            self,
            envy=   envy,
            logger= self.__log,
            **kwargs)
        self.envy = envy # INFO: type "upgrade" for pycharm editor
        self.num_of_actions = self.envy.num_actions()
        self.__log.info(f'*** FATrainer initialized, number of actions: {self.num_of_actions}')

    # selects 100% random action from action space
    def get_exploring_action(self):
        return np.random.choice(self.num_of_actions)