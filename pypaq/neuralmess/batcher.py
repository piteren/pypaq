"""

 2020 (c) piteren

    NN Batcher

"""

import numpy as np

from pypaq.lipytools.stats import msmx


BTYPES = [
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
            btype: str=         'random_cov',
            seed=               123,
            verb=               0):

        self.verb = verb
        self.seed_counter = seed

        assert btype in BTYPES, f'ERR: unknown btype, possible: {BTYPES}'
        self.btype = btype

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


# test for coverage of batchers
def test_coverage(
        btype,
        num_samples=    1000,
        batch_size=     64,
        num_batches=    1000):

    print(f'\nStarts coverage of {btype}')

    samples = np.arange(num_samples)
    np.random.shuffle(samples)
    samples = samples.tolist()

    labels = np.random.choice(2, num_samples).tolist()

    data = {
        'samples':  samples,
        'labels':   labels}

    batcher = Batcher(data, batch_size, btype=btype, verb=1)

    sL = []
    n_b = 0
    s_counter = {s: 0 for s in range(num_samples)}
    for _ in range(num_batches):
        sL += batcher.get_batch()['samples'].tolist()
        n_b += 1
        if len(set(sL)) == num_samples:
            print(f'got full coverage with {n_b} batches')
            for s in sL: s_counter[s] += 1
            sL = []
            n_b = 0

    print(msmx(list(s_counter.values()))['string'])

    print(f' *** finished coverage tests')

# test for batcher reproducibility with seed
def test_seed():

    print(f'\nStarts seed tests')

    c_size = 1000
    b_size = 64

    samples = np.arange(c_size)
    np.random.shuffle(samples)
    samples = samples.tolist()

    labels = np.random.choice(2, c_size).tolist()

    data = {
        'samples': samples,
        'labels': labels}

    batcher = Batcher(data, b_size, btype='random_cov', verb=1)
    sA = []
    while len(sA) < 10000:
        sA += batcher.get_batch()['samples'].tolist()
        np.random.seed(len(sA))

    batcher = Batcher(data, b_size, btype='random_cov', verb=1)
    sB = []
    while len(sB) < 10000:
        sB += batcher.get_batch()['samples'].tolist()
        np.random.seed(10000000-len(sB))

    seed_is_fixed = True
    for ix in range(len(sA)):
        if sA[ix] != sB[ix]: seed_is_fixed = False
    print(f'final result: seed is fixed: {seed_is_fixed}!')
    print(f' *** finished seed tests')


if __name__ == '__main__':

    for cov in BTYPES: test_coverage(cov)
    test_seed()