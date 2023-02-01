from typing import Optional

import numpy as np
import torch

from pypaq.lipytools.pylogger import get_pylogger


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

        if not logger: logger = get_pylogger()
        self._log = logger

        self._step = starting_step
        self.warm_up = warm_up or 0
        self.ann_base = ann_base
        self.ann_step = ann_step
        self.n_wup_off = n_wup_off

        super(ScaledLR, self).__init__(optimizer, last_epoch, verbose=self._log.getEffectiveLevel()<20)

    # updates LR of 0 group
    def update_base_lr0(self, lr: float):
        self.base_lrs[0] = lr

    def get_lr(self):

        lrs = np.asarray(self.base_lrs) # self.base_lrs keeps [baseLR] of groups
        if self.warm_up:
            wm_ratio = min(self._step, self.warm_up) / self.warm_up
            lrs *= wm_ratio
            self._log.debug(f'applied warmUp ({self.warm_up}) to lR')

        if self.ann_base is not None and self.ann_base != 1.0:
            steps_offs = max(0, self._step - int(self.warm_up * self.n_wup_off))
            lrs *= self.ann_base ** (steps_offs * self.ann_step)
            self._log.debug(f'applied annealing to lR ({self.ann_base:.5f},{self.ann_step:.5f})')

        self._log.debug(f'ScaledLR scheduler step: {self._step} lrs: {lrs.tolist()}')
        self._step += 1
        return lrs.tolist()

    def _get_closed_form_lr(self):
        return self.get_lr()
