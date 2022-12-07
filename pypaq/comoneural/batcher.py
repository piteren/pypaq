"""

 2020 (c) piteren

    NN Batcher
        takes data and prepares batches
        data for training, validation or testing is a dict: {key: np.array or torch.tensor}
        batch is prepared from each key

"""

import numpy as np

from pypaq.lipytools.pylogger import get_pylogger


BATCHING_TYPES = [
    'base',         # prepares batches in order of given data
    'random',       # basic random sampling
    'random_cov']   # random sampling with full coverage of data


class BatcherException(Exception):
    pass


class Batcher:

    def __init__(
            self,
            data_TR: dict,
            data_VL: dict=      None,
            data_TS: dict=      None,
            batch_size: int=    16,
            bs_mul: int=        2,      # VL & TS batch_size multiplier
            batching_type: str= 'random_cov',
            seed=               123,
            logger=             None):

        self.__log = logger or get_pylogger()

        self.seed_counter = seed

        if batching_type not in BATCHING_TYPES:
            raise BatcherException('ERR: unknown batching_type!')

        self.btype = batching_type

        self._data_keys = sorted(list(data_TR.keys()))

        self._data_TR = data_TR
        self._data_VL = data_VL
        self._data_TS = data_TS
        self._data_len_TR = self._data_TR[self._data_keys[0]].shape[0]

        self._batch_size = None
        self.set_batch_size(batch_size)
        self._bs_mul = bs_mul

        self._data_ixmap = []

        self._VL_batches = None
        self._TS_batches = None

        self.__log.info(f'*** Batcher *** initialized with {self._data_len_TR} samples of data in keys, batch size: {batch_size}')
        self.__log.debug('> Batcher keys:')
        for k in self._data_keys:
            self.__log.debug(f'>> {k}, shape: {self._data_TR[k].shape}, type:{type(self._data_TR[k][0])}')


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
        if bs > self._data_len_TR:
            raise BatcherException('ERR: cannot set batch size larger than given TR data!')
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
        if self._VL_batches is None:
            if self._data_VL is None:
                raise BatcherException('ERR: cannot prepare VL batches - data nat given')
            self._VL_batches =  Batcher.__split_data(self._data_VL, self._batch_size * self._bs_mul)
        return self._VL_batches


    def get_TS_batches(self) -> list:
        if self._TS_batches is None:
            if self._data_TS is None:
                err = 'ERR: cannot prepare TS batches - data nat given'
                self.__log.error(err)
                raise BatcherException(err)
            self._TS_batches = Batcher.__split_data(self._data_TS, self._batch_size * self._bs_mul)
        return self._TS_batches