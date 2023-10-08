import time
from typing import Dict

# time reporting helper
class TimeRep:

    def __init__(self):
        self.tr: Dict[str,float] = {}
        self.stime: float = time.time()

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