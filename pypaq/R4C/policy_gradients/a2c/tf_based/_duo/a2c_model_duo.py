from abc import abstractmethod, ABC
import numpy as np


from pypaq.lipytools.little_methods import stamp
from pypaq.lipytools.plots import two_dim_multi
from pypaq.R4C.helpers import extract_from_batch
from pypaq.R4C.actor import Actor
from pypaq.R4C.policy_gradients.a2c.tf_based._duo.a2c_graph_duo import a2c_graph_duo
from pypaq.neuralmess_duo.nemodelduo import NEModelDUO


class A2CModel_duo(Actor, ABC):

    def __init__(
            self,
            observation,
            num_actions: int,
            mdict :dict,
            graph=          a2c_graph_duo,
            devices=        -1,
            hpmser_mode=    False,
            verb=           1):

        self.verb = verb

        self.num_actions = num_actions
        observation_width = self.observation_vec(observation).shape[-1]

        if self.verb>0: print(f'\n*** A2CModel_duo inits, observation_width: {observation_width}')

        mdict['observation_width'] = observation_width
        mdict['num_actions'] = self.num_actions
        if 'name' not in mdict: mdict['name'] = f'a2c_duo_{stamp()}'

        self.nn = NEModelDUO(
            fwd_func=       graph,
            devices=        devices,
            hpmser_mode=    hpmser_mode,
            save_topdir=    '_models',
            #verb=           self.verb-1,
            verb=           self.verb,
            **mdict)

        self.nn.build_callable_submodel(
            name=       'probs_model',
            inputs=     'observation',
            outputs=    'action_prob')

        self.__upd_step = 0

    # returns vector of feats from given observation (encodes observation into input vector)
    @abstractmethod
    def observation_vec(self, observation) -> np.ndarray: pass

    # vectorization of observations batch, may be overridden with more optimal custom implementation
    def observation_vec_batch(self, observations) -> np.ndarray:
        return np.array([self.observation_vec(v) for v in observations])

    # USED for step by step actions of Actor
    def get_policy_probs(self, observation) -> np.ndarray:
        ov = self.observation_vec(observation)
        ov = np.array([ov])
        out = self.nn.call(
            data=   {'observation': ov},
            name=   'probs_model')
        probs = out['action_prob']
        return probs.numpy()[0]

    def get_policy_probs_batch(self, observations) -> np.ndarray:
        ovs = self.observation_vec_batch(observations)
        out = self.nn.call(
            data=   {'observation': ovs},
            name=   'probs_model')
        probs = out['action_prob']
        return probs.numpy()

    def get_policy_action(self, observation, sampled=False) -> int:
        probs = self.get_policy_probs(observation)
        if sampled: action = np.random.choice(self.num_actions, p=probs)
        else:       action = int(np.argmax(probs))
        return action

    def update_with_experience(self, batch, inspect=False) -> float:

        observations =  extract_from_batch(batch, 'observation')
        actions=        extract_from_batch(batch, 'action')
        dreturns =      extract_from_batch(batch, 'dreturn')
        #dreturns_norm = zscore_norm(dreturns)

        data = {
            'observation':  self.observation_vec_batch(observations),
            'action':       actions,
            'ret':          dreturns, #dreturns_norm # TODO: here we use not normalized dreturns
        }

        out = self.nn.train_batch(data=data)

        self.__upd_step += 1

        if self.verb>0:
            self.nn.log_TB(out['loss'],             'upd/loss',             step=self.__upd_step)
            self.nn.log_TB(out['loss_actor'],       'upd/loss_actor',       step=self.__upd_step)
            self.nn.log_TB(out['loss_critic'],      'upd/loss_critic',      step=self.__upd_step)
            self.nn.log_TB(out['ggnorm'],           'upd/gn',               step=self.__upd_step)
            self.nn.log_TB(out['ggnorm_avt'],       'upd/gn_avt',           step=self.__upd_step)
            self.nn.log_TB(out['amax_prob'],        'upd/amax_prob',        step=self.__upd_step)
            self.nn.log_TB(out['amin_prob'],        'upd/amin_prob',        step=self.__upd_step)
            self.nn.log_TB(out['actor_ce_mean'],    'upd/actor_ce_mean',    step=self.__upd_step)

        if inspect:
            #print(f'\nBatch size: {len(batch)}')
            #print(f'observations: {observations.shape}, first: {observations[0]}')
            #print(f'actions: {actions}')
            #print(f'rewards: {rewards.shape}, first: {rewards[0]}')
            #print(f'dreturns: {dreturns.shape}, first: {dreturns[0]}')
            #print(f'action_prob: {out["action_prob"]}')
            #print(f'action_prob_selected: {out["action_prob_selected"]}')
            #print(f'actor_ce: {out["actor_ce"]}')
            rewards = extract_from_batch(batch, 'reward')
            two_dim_multi(
                ys=         [rewards, dreturns,
                             #dreturns_norm,
                             out['advantage'], out['value']],
                names=      ['rewards', 'dreturns',
                             #'dreturns_norm',
                             'advantage', 'value'],
                legend_loc= 'lower left')

        return out['loss_actor']

    def save(self):
        self.nn.save()

if __name__ == '__main__':

    class CP_A2CModel(A2CModel_duo):

        def observation_vec(self, observation) -> np.ndarray:
            return np.array(observation)

        def observation_vec_batch(self, observations) -> np.ndarray:
            return np.array(observations)

    from applied_RL.other.cart_pole_envy import CartPoleEnvy

    envy = CartPoleEnvy(
        reward_scale=   0.1,
        lost_penalty=   0,
        verb=           1)

    mdict = {
        'hidden_layers':    (20,20),
        #'lay_norm':         True,
        'baseLR':              0.01,
        #'do_clip':          True,
        'seed':             121}

    model = CP_A2CModel(
        observation=    envy.get_observation(),
        num_actions=    4,
        mdict=          mdict,
        graph=          a2c_graph_duo
    )