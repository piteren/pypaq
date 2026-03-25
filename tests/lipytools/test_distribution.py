import numpy as np

from pypaq.lipytools.distribution import sample_to_flatten
from pypaq.lipytools.plots import histogram


def test_sample_to_flatten():

    vals = np.random.normal(size=100000)
    histogram(vals)

    vals_re = sample_to_flatten(vals)
    histogram(vals_re)

    vals_conc = np.concatenate((vals, vals_re))
    histogram(vals_conc)

    vals_re = sample_to_flatten(vals, bins=5, n_new=50000)
    histogram(vals_re)

    vals_conc = np.concatenate((vals, vals_re))
    histogram(vals_conc)