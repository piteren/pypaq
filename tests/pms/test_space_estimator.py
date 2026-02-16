import math
import numpy as np
import random
import time

from pypaq.pms.space_estimator import RBFRegressor


def test_base():

    def func(d:np.ndarray, rand_range:float=2.0):
       return np.asarray([math.sin(x)+math.sin(y)+rand_range*random.random() for x,y in d])

    rng = 10

    n_add = 20
    n_loops = 50

    rr = RBFRegressor()

    s_time = time.time()
    for loop in range(n_loops):

        new_points = rng * np.random.rand(n_add, 2) - rng / 2
        vals_new_points = func(new_points)

        nfo = rr.update(X_new=new_points, y_new=vals_new_points)
        print(f'{loop:2} {nfo} {rr}')

    print(f'time taken: {time.time() - s_time:.1f}s')
