import random
import time
import unittest

from pypaq.lipytools.time_reporter import TimeRep


class TestTimeRep (unittest.TestCase):

    def test_base(self):

        tr = TimeRep()

        s = 0
        for ix in range(5):
            stime = random.random()*3
            s += stime
            time.sleep(stime)
            print(f'{ix} --> {stime}')
            tr.log(f'phase {ix}')
        print(s)

        print(tr)