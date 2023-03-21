import math
import numpy as np
import random
import time
import unittest

from pypaq.pms.space.space_estimator import RBFRegressor


class TestRBFRegressor(unittest.TestCase):

    def test_base(self):

        def func(d:np.ndarray, rand_range:float=2.0):
           return np.asarray([math.sin(x)+math.sin(y)+rand_range*random.random() for x,y in d])

        rng = 10

        n_add = 20
        n_loops = 20

        rr = RBFRegressor()

        s_time = time.time()
        for loop in range(n_loops):

            new_points = rng * np.random.rand(n_add, 2) - rng / 2
            vals_new_points = func(new_points)

            l = rr.update(X_new=new_points, y_new=vals_new_points)
            print(f'{loop:2} {l:.4f} {rr}')

        print(f'time taken: {time.time() - s_time:.1f}s')