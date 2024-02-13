import random
import time
import unittest

from pypaq.lipytools.time_reporter import TimeRep


class TestTimeRep (unittest.TestCase):

    def test_base(self):

        tr = TimeRep()

        for ph in range(5):
            print(f'ph:{ph}')

            sub_tr = None
            # sub-phase
            if random.random() < 0.5:
                sub_tr = TimeRep()
                for sph in range(random.randint(1,4)):
                    print(f'sph:{sph}')
                    stime = random.random()
                    time.sleep(stime)
                    sub_tr.log(f'sub-phase {ph}_{sph}')
            else:
                stime = random.random()*3
                time.sleep(stime)

            tr.log(f'phase {ph}', interval_tr=sub_tr)

        print(tr)

        print()
        rep = tr.get_report(sub=True)
        for k in rep:
            print(f'{k}: {rep[k]}')

        print()
        rep = tr.get_report(total=True)
        for k in rep:
            print(f'{k}: {rep[k]}')