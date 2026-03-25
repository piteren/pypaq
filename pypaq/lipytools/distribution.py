from pypaq.lipytools.plots import histogram
import numpy as np


def sample_to_flatten(
        vals: np.ndarray,
        n_new = None,
        bins: int = 40,
        rng = None,
        seed: int = 123,
):

    if rng is None:
        rng = np.random.default_rng(seed)

    counts, edges = np.histogram(vals, bins=bins)
    deficits = counts.max() - counts

    if n_new:
        deficits = (deficits * (n_new / deficits.sum())).astype(int)

    new_vals = []
    for ba, bb, d in zip(edges[:-1], edges[1:], deficits):
        new_vals.append(ba + rng.random(d) * (bb-ba))

    return np.concatenate(new_vals)