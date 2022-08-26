"""

 2020 (c) piteren

    NN Batcher

"""

import numpy as np


BATCHING_TYPES = [
    'base',         # prepares batches in order of given data
    'random',       # basic random sampling
    'random_cov']   # random sampling with full coverage of data


class Batcher:

    def __init__(
            self,
            data_TR: dict,              # {key: np.array or list}, batch is prepared from each key
            data_VL: dict=      None,
            data_TS: dict=      None,
            batch_size: int=    16,
            bs_mul: int=        2,      # VL & TS batch_size multiplier
            batching_type: str= 'random_cov',
            seed=               123,
            verb=               0):

        self.verb = verb
        self.seed_counter = seed

        assert batching_type in BATCHING_TYPES, f'ERR: unknown batching_type'
        self.btype = batching_type

        self._data_keys = sorted(list(data_TR.keys()))

        self._data_TR = {k: np.asarray(data_TR[k]) for k in self._data_keys} # as numpy
        self._data_VL = {k: np.asarray(data_VL[k]) for k in self._data_keys} if data_VL else None
        self._data_TS = {k: np.asarray(data_TS[k]) for k in self._data_keys} if data_TS else None
        self._data_len_TR = self._data_TR[self._data_keys[0]].shape[0]

        self._batch_size = None
        self.set_batch_size(batch_size)
        self._bs_mul = bs_mul

        self._data_ixmap = []

        if verb>0:
            print(f'\nBatcher initialized with {self._data_len_TR} samples of data in keys:')
            for k in self._data_keys: print(f' > {k}, shape: {self._data_TR[k].shape}, type:{type(self._data_TR[k][0])}')
            print(f' batch size: {batch_size}')


    def _extend_ixmap(self):

        if self.btype == 'base':
            self._data_ixmap += list(range(self._data_len_TR))

        if self.btype == 'random':
            self._data_ixmap += np.random.choice(
                a=          self._data_len_TR,
                size=       self._batch_size,
                replace=    False).tolist()

        if self.btype == 'random_cov':
            new_ixmap = np.arange(self._data_len_TR)
            np.random.shuffle(new_ixmap)
            new_ixmap = new_ixmap.tolist()
            self._data_ixmap += new_ixmap

    def set_batch_size(self, bs: int):
        assert not self._data_len_TR < bs, 'ERR: cannot set batch size larger than data!'
        self._batch_size = bs

    def get_batch(self) -> dict:

        # set seed
        np.random.seed(self.seed_counter)
        self.seed_counter += 1

        if len(self._data_ixmap) < self._batch_size: self._extend_ixmap()
        
        indexes = self._data_ixmap[:self._batch_size]
        self._data_ixmap = self._data_ixmap[self._batch_size:]
        return {k: self._data_TR[k][indexes] for k in self._data_keys}

    @staticmethod
    def __split_data(data:dict, size:int) -> list:
        split = []
        counter = 0
        keys = list(data.keys())
        while counter*size < len(data[keys[0]]):
            split.append({k: data[k][counter*size:(counter+1)*size] for k in keys})
            counter += 1
        return split

    def get_VL_batches(self) -> list:
        assert self._data_VL, 'ERR: cannot prepare VL batches - data nat given'
        return Batcher.__split_data(self._data_VL, self._batch_size * self._bs_mul)

    def get_TS_batches(self) -> list:
        assert self._data_TS, 'ERR: cannot prepare VL batches - data nat given'
        return Batcher.__split_data(self._data_TS, self._batch_size * self._bs_mul)