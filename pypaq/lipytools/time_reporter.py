import time
from typing import Dict, List, Tuple, Optional


class TimeRep:
    """ time reporting helper """

    def __init__(self):
        self.start_now()

    def start_now(self):
        """ resets and starts """
        self.tr: Dict[str, Tuple[float, Optional[TimeRep]]] = {}
        self.stime = time.time()
        self.stime_start = self.stime

    def log(self, interval_name:str, interval_tr:Optional["TimeRep"]=None):
        ct = time.time()
        self.tr[interval_name] = (ct - self.stime, interval_tr)
        self.stime = ct

    def get_report(self) -> Dict[str,float]:
        return {k: self.tr[k][0] for k in self.tr}

    @staticmethod
    def _get_line(n,t):
        return f'{n:30}: {t:9.3f}s'

    def _get_report_lines(self) -> List[str]:
        rl = []
        for n,(t,itr) in self.tr.items():
            rl.append(self._get_line(n,t))
            if itr:
                rl += [f'---{l}' for l in itr._get_report_lines()]
        return rl

    def __str__(self):
        s = "\n".join(self._get_report_lines())
        return s + f"\n{self._get_line('___total time', time.time() - self.stime_start)}"