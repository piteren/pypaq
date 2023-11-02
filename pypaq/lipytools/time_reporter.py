import time
from typing import Dict

# time reporting helper
class TimeRep:

    def __init__(self):
        self.tr: Dict[str,float] = {}
        self.stime: float = time.time()
        self.stime_init = self.stime

    # resets and starts
    def start_now(self):
        self.tr = {}
        self.stime = time.time()

    def log(self, past_interval_name:str):
        ct = time.time()
        self.tr[past_interval_name] = ct - self.stime
        self.stime = ct

    def get_report(self) -> Dict[str,float]:
        rep = {}
        rep.update(self.tr)
        return rep

    def __str__(self):
        rep = self.get_report()
        rep['___total time'] = time.time() - self.stime_init
        return "\n".join([f'{k:30}: {v:9.3f}s' for k,v in rep.items()])