from collections import OrderedDict
import torch
from typing import Optional


class TorchnessException(Exception):
    pass

# weights initializer from BERT, the only difference is that in torch values are CLAMPED not SAMPLED till in <a,b>
def bert_initializer(*args, std=0.02, **kwargs):
    return torch.nn.init.trunc_normal_(*args, **kwargs, std=std, a=-2*std, b=2*std)

def my_initializer(*args, std=0.02, **kwargs):
    # different layers use different initialization functions:
    # torch Linear % Conv1D uses kaiming_uniform_(weights) & xavier_uniform_(bias)
    # my TF uses trunc_normal_(weights, std=0.02, a==b==2*std) & 0(bias) <- from BERT
    # - kaiming_uniform_ is uniform_ with bound from 2015 paper, (for relu)
    # - xavier_uniform_ is uniform_ whit bound from 2010 paper (for linear / sigmoid)
    # - trunc_normal_ is normal with mean 0 and given std, all values SAMPLED till in <a,b>
    return bert_initializer(*args, **kwargs, std=std)

# weighted merge of two checkpoints, does NOT check for compatibility of two checkpoints, but will crash if those are not compatible
def mrg_ckpts(
        ckptA: str,                     # checkpoint A (file name)
        ckptB: Optional[str],           # checkpoint B (file name), for None takes 100% ckptA
        ckptM: str,                     # checkpoint merged (file name)
        ratio: float=           0.5,    # ratio of merge
        noise: float=           0.0     # noise factor, amount of noise added to new value <0.0;1.0>
):

    checkpoint_A = torch.load(ckptA)
    checkpoint_B = torch.load(ckptB) if ckptB else checkpoint_A

    cmsd_A = checkpoint_A['model_state_dict']
    cmsd_B = checkpoint_B['model_state_dict']
    cmsd_M = OrderedDict()

    # TODO: watch out for not-float-tensors -> such should be taken from A without a mix
    for k in cmsd_A:
        std_dev = float(torch.std(cmsd_A[k]))
        noise_tensor = torch.zeros_like(cmsd_A[k])
        if std_dev != 0.0: # bias variable case
            my_initializer(noise_tensor, std=std_dev)
        cmsd_M[k] = ratio * cmsd_A[k] + (1 - ratio) * cmsd_B[k] + noise * noise_tensor

    checkpoint_M = {}
    checkpoint_M.update(checkpoint_A)
    checkpoint_M['model_state_dict'] = cmsd_M

    torch.save(checkpoint_M, ckptM)


def scaled_cross_entropy(
        labels,
        scale,
        logits: Optional[torch.Tensor]= None,
        probs: Optional[torch.Tensor]=  None) -> torch.Tensor:

    if logits is None and probs is None:
        raise TorchnessException('logits OR probs must be given!')

    if probs is None:
        probs = torch.nn.functional.softmax(input=logits, dim=-1)

    prob_label = probs[range(len(labels)), labels] # probability of class from label

    # merge loss for positive and negative advantage
    ce = torch.where(
        condition=  scale > 0,
        input=      -torch.log(prob_label),
        other=      -torch.log(1-prob_label))

    return ce * torch.abs(scale)