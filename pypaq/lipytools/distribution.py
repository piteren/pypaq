import numpy as np
from typing import Optional

from pypaq.lipytools.plots import histogram


def sample_to_flatten(
        vals: np.ndarray,
        n_new: Optional[int] = None,
        bins: int = 40,
        rng: Optional[np.random.Generator] = None,
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