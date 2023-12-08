import unittest

from pypaq.pms.paspa import PaSpa
from pypaq.pms.points_cloud import VPoint, PointsCloud

SAMPLE_PSDD = {
    'a':    [0.0,   1.0],
    'b':    [-5.0,  5],
    'c':    [0,     10],
    'd':    [-10,   -5],
    'e':    (-1,-7,10,15.5,90,30),
    'f':    (1,8),
    'g':    (-11,-2,-3,4,5,8,9),
    'h':    (True, False, None),
    'i':    ('tat', 'mam', 'kot'),
    'j':    (6,0,1,2,3,None),
    'k':    [3,     4],
    'l':    [5,     5.5],
    'm':    (8,),
    'n':    (False,),
    'o':    (None,),
    'p':    (1,2,3,4,5,6,7,8,9,10.5,-11,-4.4),
    'r':    [-3,    -4],
    's':    (0,0.1),
    't':    (1,2,3,4,8,4,'','wert','5')}


class TestVPoint(unittest.TestCase):

    def test_base(self):
        paspa = PaSpa(SAMPLE_PSDD)
        point = paspa.sample_point()
        vpoint = VPoint(point=point, name='test', value=0.1)
        print(vpoint)

class TestPointCloud(unittest.TestCase):

    def test_base(self):
        paspa = PaSpa(SAMPLE_PSDD)
        points = [paspa.sample_point() for _ in range(20)]
        pcloud = PointsCloud(paspa=paspa)
        pcloud.update_cloud([VPoint(point=p, name=f'p{ix:02}') for ix,p in enumerate(points)])
        print(pcloud)
