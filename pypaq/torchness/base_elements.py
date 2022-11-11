from collections import OrderedDict
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from pypaq.lipytools.pylogger import get_pylogger

def my_initializer(*args, std=0.02, **kwargs):
    # https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
    return torch.nn.init.trunc_normal_(*args, **kwargs, std=std)

# applies warm-up at the beginning (for warm_up steps) and annealing after some steps (after warm_up * n_wup_off), should be called every step
class ScaledLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
            self,
            optimizer,
            starting_step: int=         0,      # step to start with
            warm_up: Optional[int]=     1000,   # number of steps for linear warm-up, for None or 0 warm-up is turned off
            ann_base: Optional[float]=  0.999,  # annealing base, None or 1.0 turns off annealing
            ann_step: float=            1.0,    # annealing step, higher value speeds up annealing
            n_wup_off: float=           2.0,    # number of warm-up durations to start annealing
            last_epoch=                 -1,
            logger=                     None):

        if not logger: logger = get_pylogger(name='ScaledLR')
        self.__log = logger

        self._step = starting_step
        self.warm_up = warm_up or 0
        self.ann_base = ann_base
        self.ann_step = ann_step
        self.n_wup_off = n_wup_off

        super(ScaledLR, self).__init__(optimizer, last_epoch, verbose=self.__log.getEffectiveLevel()<20)

    # updates LR of 0 group
    def update_base_lr0(self, lr: float):
        self.base_lrs[0] = lr

    def get_lr(self):

        lrs = np.array(self.base_lrs) # self.base_lrs keeps [baseLR] of groups
        if self.warm_up:
            wm_ratio = min(self._step, self.warm_up) / self.warm_up
            lrs *= wm_ratio
            self.__log.debug(f'applied warmUp ({self.warm_up}) to lR')

        if self.ann_base is not None and self.ann_base != 1.0:
            steps_offs = max(0, self._step - self.warm_up * self.n_wup_off)
            lrs *= self.ann_base ** (steps_offs * self.ann_step)
            self.__log.debug(f'applied annealing to lR ({self.ann_base:.5f},{self.ann_step:.5f})')

        self.__log.debug(f'ScaledLR scheduler step: {self._step} lrs: {lrs.tolist()}')
        self._step += 1
        return lrs.tolist()

    def _get_closed_form_lr(self):
        return self.get_lr()

# copied & refactored from torch.nn.utils.clip_grad.py, used by GradClipperAVT
def clip_grad_norm_(
        parameters,
        max_norm: float,
        norm_type: float=   2.0,
        do_clip: bool=      True    # disables clipping (just GN calculations)
) -> float:

    if isinstance(parameters, torch.Tensor): parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0: return 0.0

    device = parameters[0].grad.device
    if norm_type == torch._six.inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

    if do_clip:
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for p in parameters:
            p.grad.detach().mul_(clip_coef_clamped.to(p.grad.device))

    return float(total_norm.cpu().numpy())

# gradient clipping class, clips gradients by clip_value or gg_avt_norm (computed with averaging window)
class GradClipperAVT:

    def __init__(
            self,
            module: torch.nn.Module,
            clip_value: Optional[float]=    None,   # clipping value, for None clips with avt
            avt_SVal: float=                0.1,    # start value for AVT (smaller value makes gradients warm-up)
            avt_window: int=                100,    # width of averaging window (number of steps)
            avt_max_upd: float=             1.5,    # max factor of gg_avt_norm to update with
            do_clip: bool=                  True,   # disables clipping (just GN calculations)
            logger=                         None):

        if not logger: logger = get_pylogger(name='ScaledLR')
        self.__log = logger

        self.module = module
        self.clip_value = clip_value
        self.gg_avt_norm = avt_SVal
        self.avt_window = avt_window
        self.avt_max_upd = avt_max_upd
        self.do_clip = do_clip


    def clip(self):

        gg_norm = clip_grad_norm_(
            parameters= self.module.parameters(),
            max_norm=   self.clip_value or self.gg_avt_norm,
            do_clip=    self.do_clip)

        # in case of gg_norm explodes we want to update self.gg_avt_norm with value of self.avt_max_upd * self.gg_avt_norm
        avt_update = min(gg_norm, self.avt_max_upd * self.gg_avt_norm)
        self.gg_avt_norm = (self.gg_avt_norm * (self.avt_window-1) + avt_update) / self.avt_window # update
        self.__log.debug(f'clipped with: gg_avt_norm({self.gg_avt_norm})')

        return {
            'gg_norm':      gg_norm,
            'gg_avt_norm':  self.gg_avt_norm}

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
            torch.nn.init.trunc_normal_(
                tensor= noise_tensor,
                std=    std_dev,
                a=      -2*std_dev,
                b=      2 *std_dev)
        cmsd_M[k] = ratio * cmsd_A[k] + (1 - ratio) * cmsd_B[k] + noise * noise_tensor

    checkpoint_M = {}
    checkpoint_M.update(checkpoint_A)
    checkpoint_M['model_state_dict'] = cmsd_M
    torch.save(checkpoint_M, ckptM)

# TensorBoard writer
class TBwr:

    def __init__(
            self,
            logdir: str,
            flush_secs= 10):
        self.logdir = logdir
        self.flush_secs = flush_secs
        # INFO: SummaryWriter creates logdir while init, because of that self.sw init has moved here (in the first call of add)
        self.sw = None

    def add(self,
            value,
            tag: str,
            step: int):

        if not self.sw:
            self.sw = SummaryWriter(
                log_dir=    self.logdir,
                flush_secs= self.flush_secs)

        self.sw.add_scalar(tag, value, step)

    def flush(self): self.sw.flush()