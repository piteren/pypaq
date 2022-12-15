from typing import Optional

import torch

from pypaq.lipytools.pylogger import get_pylogger


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
        self._log = logger

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
        self._log.debug(f'clipped with: gg_avt_norm({self.gg_avt_norm})')

        return {
            'gg_norm':      gg_norm,
            'gg_avt_norm':  self.gg_avt_norm}
