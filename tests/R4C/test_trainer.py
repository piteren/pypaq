import unittest

from pypaq.R4C.trainer import ExperienceMemory


class TestExperienceMemory(unittest.TestCase):

    def test_ExperienceMemory(self):

        mem = ExperienceMemory(maxsize=20)

        for n in range(30):
            e = {'n': n}
            mem.append(e)

        print(mem.get_all())
        print(mem.get_all())
        s = mem.sample(20)
        print(s)
        sn = [e['n'] for e in s]
        print(len(set(sn)))