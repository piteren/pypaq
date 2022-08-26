import unittest
import numpy as np

from pypaq.neuralmess_duo.batcher import Batcher, BATCHING_TYPES
from pypaq.lipytools.stats import msmx


class TestBatcher(unittest.TestCase):

    def test_coverage(
            self,
            num_samples=    1000,
            batch_size=     64,
            num_batches=    1000):

        for btype in BATCHING_TYPES:
            print(f'\nStarts coverage of {btype}')

            samples = np.arange(num_samples)
            np.random.shuffle(samples)
            samples = samples.tolist()

            labels = np.random.choice(2, num_samples).tolist()

            data = {
                'samples':  samples,
                'labels':   labels}

            batcher = Batcher(data, batch_size=batch_size, batching_type=btype, verb=1)

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

    # test for Batcher reproducibility with seed
    def test_seed(self):

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

        batcher = Batcher(data, batch_size=b_size, batching_type='random_cov', verb=1)
        sA = []
        while len(sA) < 10000:
            sA += batcher.get_batch()['samples'].tolist()
            np.random.seed(len(sA))

        batcher = Batcher(data, batch_size=b_size, batching_type='random_cov', verb=1)
        sB = []
        while len(sB) < 10000:
            sB += batcher.get_batch()['samples'].tolist()
            np.random.seed(10000000-len(sB))

        seed_is_fixed = True
        for ix in range(len(sA)):
            if sA[ix] != sB[ix]: seed_is_fixed = False
        print(f'final result: seed is fixed: {seed_is_fixed}!')
        print(f' *** finished seed tests')