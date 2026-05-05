import time

from pypaq.exception import PyPaqException


class TimeRep:
    """ time reporting helper """

    def __init__(self):
        self.start_now()

    def start_now(self):
        """ resets and starts """
        self.tr: dict[str, tuple[float, "TimeRep | None"]] = {}
        self.stime = time.time()
        self.stime_start = self.stime
        self.sub_tr_keys = []

    def log(self, interval_name: str, interval_tr: "TimeRep | None" = None):
        ct = time.time()
        if interval_name in self.tr:
            raise PyPaqException(f'interval_name {interval_name} already reported')
        if interval_tr:
            irep = interval_tr.get_report(sub=True)
            for k in irep:
                if k in self.sub_tr_keys:
                    raise PyPaqException(f'sub interval_name {k} already reported')
                self.sub_tr_keys.append(k)
        self.tr[interval_name] = (ct - self.stime, interval_tr)
        self.stime = ct

    def get_report(self, sub: bool = False, total: bool = False) -> dict[str, float]:
        rep = {}
        for k in self.tr:
            rep[k] = self.tr[k][0]
            if sub and self.tr[k][1]:
                sub_rep = self.tr[k][1].get_report(sub=True)
                for sk in sub_rep:
                    rep[f'---{sk}'] = sub_rep[sk]
        if total:
            rep['___total time'] = time.time() - self.stime_start
        return rep

    @staticmethod
    def _get_line(n, t):
        return f'{n:30}: {t:9.3f}s'

    def _get_report_lines(self) -> list[str]:
        rl = []
        for n,(t,itr) in self.tr.items():
            rl.append(self._get_line(n,t))
            if itr:
                rl += [f'---{l}' for l in itr._get_report_lines()]
        return rl

    def __str__(self):
        s = "\n".join(self._get_report_lines())
        return s + f"\n{self._get_line('___total time', time.time() - self.stime_start)}"