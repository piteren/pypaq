import numpy as np

# computes average of max and min probabilities for given array (batch) of probabilities
def avg_probs(probs: np.ndarray) -> dict:
    probs = np.reshape(probs, (-1,probs.shape[-1]))
    max_probs = np.max(probs, axis=-1)  # max action_probs
    min_probs = np.min(probs, axis=-1)  # min action_probs
    amax_prob = np.mean(max_probs)      # average of batch max action_prob
    amin_prob = np.mean(min_probs)      # average of batch min action_prob
    return {
        'amax_prob':    amax_prob,
        'amin_prob':    amin_prob}