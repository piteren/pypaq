from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
import scipy
from typing import List, Optional, Union
import warnings

from pypaq.pytypes import NPL
from pypaq.lipytools.files import prep_folder
from pypaq.lipytools.stats import msmx



def histogram(
        val_list: NPL,
        name=                       'values',
        rem_nstd: float=            0.0,    # removes values out of N*stddev
        msmx_stats=                 True,   # prints minimal stats
        density=                    True,
        bins: Optional[int]=        None,   # automatic for None
        add_density_curve: bool=    True,
        save_FD :str=               None,
) -> str:

    if type(val_list) is not np.ndarray:
        val_list = np.asarray(val_list)

    if not len(val_list):
        msg = f'cannot prepare histogram for empty val_list!'
        warnings.warn(msg)
        return msg

    val_set = list(set(val_list))

    small_int_case = False
    if len(val_set) < 50:
        val_set = [v.item() for v in val_set]
        if all([type(v) is int for v in val_set]):
            val_set = sorted(val_set)
            if val_set[-1]-val_set[0] < 60:
                small_int_case = True
    print(small_int_case)

    s = []
    if msmx_stats:
        s.append(f'histogram: "{name}" (samples: {len(val_list)}): {msmx(val_list)["string"]}')

    if rem_nstd:
        std = val_list.std()
        mean = val_list.mean()
        val_list = val_list[val_list > mean - rem_nstd * std]
        val_list = val_list[val_list < mean + rem_nstd * std]

    if not bins:
        if small_int_case:
            bins = np.arange(val_set[0]-0.5, val_set[-1]+1.5, 1)
        else:
            bins = len(val_set)
            if bins > 50: bins = 50

    plt.figure()
    n, x, _ = plt.hist(val_list, label=name, density=density, bins=bins, alpha=0.5)
    if small_int_case:
        plt.xticks(range(val_set[0], val_set[-1]+1))

    if add_density_curve:
        # try build density, for some val_list it is not possible
        try:
            density = scipy.stats.gaussian_kde(val_list)
            plt.plot(x, density(x))
        except: pass

    plt.legend(loc='upper right')
    plt.grid(True)
    if save_FD:
        prep_folder(save_FD)
        plt.savefig(f'{save_FD}/{name}.png')
    else:
        plt.show()

    plt.close()
    return '\n'.join(s)


def two_dim(
        y: Union[List,np.ndarray],                  # two(yx) or one(y) dim list or np.array
        x: Optional[Union[List,np.ndarray]]=    None,
        name=                                   'values',
        plot_type: str=                         'plot', # pot, bar, scatter
        save_FD: str =                          None,
        xlogscale=                              False,
        ylogscale=                              False,
        legend_loc=                             'upper left',
        **plot_f_kwargs): # like for scatter: alpha=0.5, s=10

    _plot_type_function = {
        'plot':     plt.plot,
        'bar':      plt.bar,
        'scatter':  plt.scatter}

    if type(y) is list: y = np.asarray(y)
    if x is None:
        if len(y.shape) < 2: x = np.arange(len(y))
        else:
            x = y[:, 1]
            y = y[:, 0]

    plt.figure()
    plot_f = _plot_type_function[plot_type]
    plot_f(x, y, label=name, **plot_f_kwargs)
    if xlogscale: plt.xscale('log')
    if ylogscale: plt.yscale('log')
    plt.legend(loc=legend_loc)
    plt.grid(True)
    if save_FD:
        prep_folder(save_FD)
        plt.savefig(f'{save_FD}/{name}.png')
    else:
        plt.show()
    plt.close()


def two_dim_multi(
        ys: List[Union[List,np.ndarray]],
        names: Optional[List[str]]= None,
        name: Optional[str]=        None,
        save_FD: str=               None,
        xlogscale=                  False,
        ylogscale=                  False,
        legend_loc=                 'upper left'):

    if names is None:
        names = [f'values_{ix}' for ix in range(len(ys))]

    x = np.arange(len(ys[0]))

    plt.figure()
    for y,name in zip(ys,names):
        plt.plot(x, y, label=name)

    plt.legend(loc=legend_loc)
    plt.grid(True)
    if xlogscale: plt.xscale('log')
    if ylogscale: plt.yscale('log')

    if save_FD:
        prep_folder(save_FD)
        if name is None:
            name = "_".join(names)
        plt.savefig(f'{save_FD}/{name}.png')
    else:
        plt.show()
    plt.close()


def three_dim(
    xyz: Union[list, np.ndarray], # sequence of (x,y,val) or (x,y,z,val)
    name=               'values',
    x_name=             'x',
    y_name=             'y',
    z_name=             'z',
    val_name=           'val',
    opacity=            0.7,
    width=              700,
    height=             700,
    save_FD: str =      None):

    # expand to 3 axes + val (3rd axis data)
    if len(xyz[0])<4:
        new_xyz = []
        for e in xyz: new_xyz.append(list(e) + [e[-1]])
        xyz = new_xyz

    df = pd.DataFrame(
        data=       xyz,
        columns=    [x_name,y_name,z_name,val_name])

    std = df[val_name].std()
    mean = df[val_name].mean()
    off = 2*std
    cr_min = mean - off
    cr_max = mean + off

    fig = px.scatter_3d(
        data_frame=     df,
        title=          name,
        x=              x_name,
        y=              y_name,
        z=              z_name,
        color=          val_name,
        range_color=    [cr_min,cr_max],
        opacity=        opacity,
        width=          width,
        height=         height)

    if save_FD:
        file = f'{save_FD}/{name}_3Dplot.html'
        fig.write_html(file, auto_open=False if os.path.isfile(file) else True)
    else:
        fig.show()
    plt.close()