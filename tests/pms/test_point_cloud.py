from pypaq.pms.paspa import PaSpa
from pypaq.pms.points_cloud import VPoint, PointsCloud, points_nice_table

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


def test_VPoint_base():
    paspa = PaSpa(SAMPLE_PSDD)
    point = paspa.sample_point()
    vpoint = VPoint(point=point, name='test', value=0.1)
    print(vpoint)


def test_VPoint_str():
    point = {'a': 1.0, 'b': 2.0}
    vp = VPoint(point=point, name='test', value=0.5)
    s = str(vp)
    assert 'test' in s

    vp_no_name = VPoint(point=point, id=42)
    s = str(vp_no_name)
    assert '42' in s


def test_PointCloud_base():
    paspa = PaSpa(SAMPLE_PSDD)
    points = [paspa.sample_point() for _ in range(20)]
    pcloud = PointsCloud(paspa=paspa)
    pcloud.update_cloud([VPoint(point=p, name=f'p{ix:02}') for ix,p in enumerate(points)])
    print(pcloud)


def test_PointCloud_distance():
    psdd = {'a': [0.0, 1.0], 'b': [0.0, 1.0]}
    paspa = PaSpa(psdd)
    pcloud = PointsCloud(paspa=paspa)

    vpa = VPoint(point={'a': 0.0, 'b': 0.0}, name='pa')
    vpb = VPoint(point={'a': 1.0, 'b': 1.0}, name='pb')

    pcloud.update_cloud([vpa, vpb])
    dist = pcloud.distance(vpa, vpb)
    assert dist > 0


def test_PointCloud_single_vpoint():
    paspa = PaSpa(SAMPLE_PSDD)
    pcloud = PointsCloud(paspa=paspa)
    point = paspa.sample_point()
    vp = VPoint(point=point, name='single')
    pcloud.update_cloud(vp)  # single VPoint, not list
    assert len(pcloud) == 1


def test_points_nice_table():
    psdd = {'a': [0.0, 1.0], 'b': [0.0, 1.0]}
    paspa = PaSpa(psdd)
    vpoints = [
        VPoint(point={'a': 0.1, 'b': 0.2}, name='p1', value=0.5),
        VPoint(point={'a': 0.9, 'b': 0.8}, name='p2', value=0.9),
    ]
    table = points_nice_table(vpoints)
    assert len(table) == 3  # header + 2 rows
    print('\n'.join(table))
