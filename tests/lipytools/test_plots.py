import numpy as np
import pandas as pd
import unittest

from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.plots import histogram, two_dim, three_dim, two_dim_multi, week_density_plot

from tests.envy import flush_tmp_dir

PLOTS_FD = f'{flush_tmp_dir()}/hpmser'


class TestPlots(unittest.TestCase):

    def setUp(self) -> None:
        prep_folder(PLOTS_FD, flush_non_empty=True)


    def test_histogram(self):
        histogram([1, 4, 5, 5, 3, 3, 5, 65, 32, 45, 5, 5, 6, 33, 5])


    def test_two_dim(self):
        two_dim([0,2,3,4,6,4,7,2,6,3,4])


    def test_two_dim_more(self):

        n = 100

        x = (np.arange(n) - n // 2) / (n / 2 / np.pi / 3)
        y = np.sin(x)

        two_dim(y, x)
        two_dim(list(zip(y, x)))
        two_dim(y)


    def test_two_dim_multi_save(self):

        va = [1,2,3,1,2,3,1,2,1]
        vb = [1,2,3,4,5,4,3,2,1]
        two_dim_multi(ys=[va,vb])

        va = [22, 22, 14, 10, 12, 4, 6, 14]
        vb = [22, 22.0, 18.0, 14.0, 13.0, 8.5, 7.2, 10.6]
        two_dim_multi(
            ys=         [va, vb],
            names=      ['va', 'vb'],
            name=       'multi_save',
            save_FD=    PLOTS_FD)


    def test_three_dim(self):

        vals = np.random.random((32,64))
        print(vals.shape)

        xyz = []
        for rix in range(vals.shape[0]):
            for eix in range(vals.shape[1]):
                xyz.append([rix, eix, vals[rix, eix]])

        three_dim(xyz)


    def test_week_density_plot(self):

        for s,e in [
            ('2024-01-08', '2024-02-21'),
            ('2024-01-08', '2024-09-11'),
            ('2021-01-15', '2026-09-14'),
        ]:
            dates = pd.date_range(s,e, freq="D")
            values = np.random.rand(len(dates))
            df = pd.DataFrame({'date': dates, 'density': values})
            week_density_plot(df, bounds=[0.0, 0.2, 0.5, 0.7])
