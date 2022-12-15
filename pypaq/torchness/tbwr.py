from torch.utils.tensorboard import SummaryWriter


# TensorBoard writer based on PyTorch wraps of TensorBoard
class TBwr:

    def __init__(
            self,
            logdir: str,
            flush_secs= 10):
        self.logdir = logdir
        self.flush_secs = flush_secs
        # INFO: SummaryWriter creates logdir while init, because of that self.sw init has moved here (in the first call of add)
        self.sw = None

    def _get_sw(self):
        return SummaryWriter(
            log_dir=    self.logdir,
            flush_secs= self.flush_secs)

    def add(self,
            value,
            tag: str,
            step: int):
        if not self.sw: self.sw = self._get_sw()
        self.sw.add_scalar(
            tag=            tag,
            scalar_value=   value,
            global_step=    step)

    def add_histogram(
            self,
            values,
            tag: str,
            step: int):
        if not self.sw: self.sw = self._get_sw()
        self.sw.add_histogram(
            tag=            tag,
            values=         values,
            global_step=    step,
            bins=           "tensorflow")

    def flush(self): self.sw.flush()
