from pypaq.lipytools.plots import two_dim_multi
from pypaq.R4C.policy_gradients.a2c.a2c_model_duo import A2CModel_duo
from pypaq.R4C.trainer import FATrainer


class A2CTrainer_duo(FATrainer):

    def __init__(
            self,
            a2c_model_duo: A2CModel_duo,
            verb=           1,
            **kwargs):

        FATrainer.__init__(self, actor=a2c_model_duo, verb=verb, **kwargs)
        self.actor = a2c_model_duo # INFO: type "upgrade" for pycharm editor

        self.num_of_actions = self.envy.num_actions()

        if self.verb>0:
            print(f'\n*** A2CTrainer for {self.envy.name} initialized')
            print(f' > actions: {self.envy.num_actions()}, exploration: {self.exploration}')

    # update is performed for both: Actor and Critic
    def update_actor(self, inspect=False):

        batch = self.memory.get_all()
        observations =  self._extract_from_batch(batch, 'observation')
        actions =       self._extract_from_batch(batch, 'action')
        dreturns =      self._extract_from_batch(batch, 'dreturn')

        dreturns_norm = self.zscore_norm(dreturns)

        if inspect:
            print(f'\nBatch size: {len(batch)}')
            print(f'observations: {observations.shape}, {observations[0]}')
            print(f'dreturns: {dreturns.shape}, {dreturns[0]}')
            rewards = self._extract_from_batch(batch, 'reward')
            two_dim_multi(
                ys=         [rewards, dreturns, dreturns_norm],
                names=      ['rewards', 'dreturns', 'dreturns_norm'],
                legend_loc= 'lower left')

        loss = self.actor.update_batch(
            observations=       observations,
            actions=            actions,
            dreturns=           dreturns_norm)

        self.memory.reset()

        return loss