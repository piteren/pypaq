from pypaq.lipytools.plots import histogram
import numpy as np


def test_sample_to_flatten():
    vals = np.random.normal(size=100000)
    histogram(vals)
    vals_re = sample_to_flatten(vals, n_new=50000)
    histogram(vals_re)
    vals_conc = np.concatenate((vals, vals_re))
    histogram(vals_conc)