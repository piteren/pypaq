import numpy as np
from typing import List



# ReinforcementLearning Exception
class RLException(Exception): pass


# extracts array of data from a batch
def extract_from_batch(
        batch :List[dict],
        key: str) -> List:
    return list(map(lambda x: x[key], batch))

# normalizes x with zscore (0 mean 1 std), this is helpful for training, as rewards can vary considerably between episodes,
def zscore_norm(x):
    if len(x) < 2: return x
    return (x - np.mean(x)) / (np.std(x) + 0.00000001)

# prepares list of discounted accumulated return from [reward]
def discounted_return(
        rewards: List[float],
        discount: float) -> List[float]:
    dar = np.zeros_like(rewards)
    s = 0.0
    for i in reversed(range(len(rewards))):
        s = s * discount + rewards[i]
        dar[i] = s
    return list(dar)

# prepares list of moving_average return from [reward]
def movavg_return(
        rewards: List[float],
        factor: float           # (0.0-0.1> factor of current reward taken for update
) -> List[float]:
    mvr = np.zeros_like(rewards)
    s = rewards[-1]
    mvr[-1] = s
    for i in reversed(range(len(rewards[:-1]))):
        s = (1-factor) * s + factor * rewards[i]
        mvr[i] = s
    return list(mvr)