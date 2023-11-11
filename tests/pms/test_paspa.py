import random
import unittest

from pypaq.pms.base import point_str
from pypaq.pms.paspa import PaSpa

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



def get_subs(psdd):
    keys = list(psdd.keys())
    for _ in range(random.randint(0,len(keys)-1)):
        keys.pop(random.randint(0,len(keys)-1))
    return {k: psdd[k] for k in keys}


class TestPaspa(unittest.TestCase):

    def test_base(self):
        paspa = PaSpa(SAMPLE_PSDD)
        print(paspa)


    def test_point_normalized(self):

        psdd = {
            'a': [0.0,1.0],
            'b': [0,9],
            'c': (1,2,4),
            'd': (1,2,None,'s')}

        paspa = PaSpa(psdd)

        pa = {
            'a': 0.0,
            'b': 1,
            'c': 2,
            'd': None}

        pb = {
            'a': 0.55,
            'b': 4,
            'c': 4,
            'd': 1}

        print(paspa.is_from_space(pa))
        self.assertTrue(paspa.is_from_space(pa))
        print(paspa.is_from_space(pb))
        self.assertTrue(paspa.is_from_space(pb))

        pan = paspa.point_normalized(pa)
        print(pan)
        self.assertTrue(pan == {'a': 0.0, 'b': 0.15000000000000002, 'c': 0.5, 'd': 0.625})
        pbn = paspa.point_normalized(pb)
        print(pbn)
        self.assertTrue(pbn == {'a': 0.55, 'b': 0.45, 'c': 0.8333333333333333, 'd': 0.125})


    def test_loop(self):

        for _ in range(100):
            paspa_a = PaSpa(get_subs(SAMPLE_PSDD))
            self.assertTrue(paspa_a.rdim <= paspa_a.dim)
            paspa_b = PaSpa(get_subs(SAMPLE_PSDD))
            self.assertTrue(paspa_b.rdim <= paspa_b.dim)

            pa = paspa_a.sample_point_GX()
            pb = paspa_a.sample_point_GX(pa)
            pa = paspa_a.sample_point_GX(pa,pb)

            pc = paspa_b.sample_point_GX()
            pd = paspa_b.sample_point_GX(pc)
            pc = paspa_b.sample_point_GX(pc, pd)

            if paspa_a != paspa_b:
                self.assertRaises(Exception, paspa_a.sample_point_GX, pa, pc)
            else: print(paspa_a.axes,paspa_b.axes)


    def test_add(self):

        psdd_a = {
            'a':    [0.0,   1.0],
            'b':    (-1,-7,10,15.5,90,30),
            'c':    (1,8.0),
            'd':    (True, False, None),
            'i':    ('tat', 'mam', 'kot', None)}

        psdd_b = {
            'a':    [0.0,   1.0],
            'b':    (13,-1,10,155),
            'c':    (1,8.0),
            'd':    (None, False),
            'i':    ('tat', 'glu', None)}

        paspa_a = PaSpa(psdd_a)
        print(paspa_a)
        paspa_b = PaSpa(psdd_b)
        print(paspa_b)
        paspa_c = paspa_a + paspa_b
        print(paspa_c)


    def test_random(self):

        num_samples = 1000

        psdd = {
            'a':    [0.0,   1.0],
            'b':    [-5.0,  5],
            'c':    [0,     10],
            'd':    [-10,   -5],
            'e':    (-1,-7,10,15.5,90,30),
            'f':    (1,8),
            'g':    (-11,-2,-3,4,5,8,9),
            'h':    (True, False, None),
            'i':    ('tat', 'mam', 'kot'),
            'j':    (0, 1, 2, 3, None)}

        paspa = PaSpa(psdd)

        print(f'\n{paspa}')

        print(f'\n### Corners of space:')
        pa, pb = paspa.sample_corners()
        print(point_str(pa))
        print(point_str(pb))
        print(f'distance: {paspa.distance(pa,pb):.3f}')

        print(f'\n### Random 100 points from space:')
        points = []
        for ix in range(num_samples):
            point = paspa.sample_point_GX()
            points.append(point)
            #print(f'{ix:2d}: {point_str(point)}')

        print(f'\n### 100 points from space with ref_point and ax_dst:')
        avg_dst_fract = 0
        max_dst_fract = 0
        for ix in range(num_samples):
            point_a = points[ix]
            ax_dst = random.random()
            point_b = paspa.sample_point_GX(
                pointA=         point_a,
                prob_noise=     1,
                noise_scale=    ax_dst,
                prob_axis=      0.0,
                prob_diff_axis= 0.0)
            res_ax_dst = paspa.distance(point_a, point_b)
            dst_fract = res_ax_dst / ax_dst
            if dst_fract > max_dst_fract: max_dst_fract = dst_fract
            avg_dst_fract += dst_fract
            if res_ax_dst > ax_dst: print(f'GOT HIGHER RESULTING DISTANCE:')
            print(f'{ix:2d}: requested axis distance {ax_dst:.3f}, resulting space distance: {res_ax_dst:.3f}')
            print(f'  {point_str(point_a)}')
            print(f'  {point_str(point_b)}')
        print(avg_dst_fract/num_samples)
        print(max_dst_fract)


    def test_close_points(self):

        psd = {
            'pe_width':             [0,5],
            'pe_min_pi':            [0.05,1],
            'pe_max_pi':            [1.0,9.9],
            't_drop':               [0.0,0.1],                  
            'f_drop':               [0.0,0.2],                 
            'n_layers':             [15,25],                    
            'lay_drop':             [0.0,0.2],                  
            'ldrt_scale':           [2,6],                      
            'ldrt_drop':            [0.0,0.5],                  
            'drt_nlays':            [0,5],                      
            'drt_scale':            [2,6],                      
            'drt_drop':             [0.0,0.6],                  
            'out_drop':             [0.0,0.5],                  
            'learning_rate':        (1e-4,1e-3,5e-3),           
            'weight_f':             [0.1,9.9],                 
            'scale_f':              [1.0,6],                    
            'warm_up':              (100,200,500,1000,2000),   
            'ann_step':             (1,2,3),                   
            'n_wup_off':            [1,50]}                     

        paspa = PaSpa(psd)
        print(f'\n{paspa}')

        ref_pt = paspa.sample_point_GX()
        print(f'\nSampled reference point:\n > {point_str(ref_pt)}')

        ld = 1
        while ld > 0:
            nref_pt = paspa.sample_point_GX(pointA=ref_pt, noise_scale=0.1)
            ld = nref_pt['ldrt_drop']
            if ld < ref_pt['ldrt_drop']:
                ref_pt = nref_pt
                print(f' next point ldrt_drop: {ref_pt["ldrt_drop"]}')
        print(f'\nFinal point:\n > {point_str(ref_pt)}')