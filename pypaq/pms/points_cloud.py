import math
from typing import Sized, List, Tuple, Optional, Dict, Union

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.stats import mam
from pypaq.lipytools.plots import three_dim
from pypaq.lipytools.printout import float_to_str
from pypaq.pms.base import POINT, point_str
from pypaq.pms.paspa import PaSpa


class VPoint:
    """ Valued Point
    with unique id & float value """

    def __init__(
            self,
            point: POINT,
            name: Optional[str]=    None,
            id: Optional[int]=      None,
            value: Optional[float]= None,
    ):
        self.point = point
        self.name = name
        self.id = id
        self.value = value

    def __str__(self):
        nfo = self.name if self.name else ""
        if not nfo:
            nfo = f'#{self.id:4}' if self.id is not None else '#_'
        return f'{nfo} [val: {self.value:.8f}] {point_str(self.point)}'


class PointsCloud(Sized):
    """ Points Cloud - cloud of Valued Points """

    def __init__(
            self,
            paspa: PaSpa,   # space of this PointsCloud
            logger=     None,
            loglevel=   20):

        if not logger:
            logger = get_pylogger(name=self.__class__.__name__, level=loglevel)
        self.logger = logger
        self.logger.info('*** PointsCloud *** initializing..')

        self.paspa = paspa

        # values below are updated with each call to update_cloud()
        self._vpointsD: Dict[int, VPoint] = {}          # {id: VPoint}
        self._nearest: Dict[int, Tuple[int,float]] = {} # {id: (id,dist)}
        self.min_nearest = math.sqrt(self.paspa.dim)
        self.avg_nearest = math.sqrt(self.paspa.dim)
        self.max_nearest = math.sqrt(self.paspa.dim)

        self.prec = 8 # print precision, will be updated while adding new vpoints


    def distance(self, vpa:VPoint, vpb:VPoint) -> float:
        """ returns distance between two VPoints """
        return self.paspa.distance(vpa.point, vpb.point)


    def update_cloud(self, vpoints:Union[VPoint,List[VPoint]]):
        """ updates Cloud (self) with given VPoint / list
        (adds new to _vpoints & updates _nearest) """

        if vpoints:

            if type(vpoints) is not list:
                vpoints = [vpoints]

            for vpoint in vpoints:

                # add to _vpoints
                vp_id = len(self)
                vpoint.id = vp_id
                self._vpointsD[vp_id] = vpoint

                # update _nearest
                his_nearest = None
                his_nearest_dist = None
                for k in self._vpointsD:
                    if k != vp_id:
                        dist = self.distance(vpoint, self._vpointsD[k])
                        if his_nearest is None or dist < his_nearest_dist:
                            his_nearest = k
                            his_nearest_dist = dist
                        if k not in self._nearest or dist < self._nearest[k][1]:
                            self._nearest[k] = vp_id, dist
                if his_nearest is not None:
                    self._nearest[vp_id] = his_nearest, his_nearest_dist

                if vpoint.value > 0.01: self.prec = 4

            self.min_nearest, self.avg_nearest, self.max_nearest = mam([v[1] for v in self._nearest.values()])


    def plot(
            self,
            name: str=                  'PointsCloud',
            axes: Optional[List[str]]=  None,   # list with axes names, 2-3, like ['drop_a','drop_b','loss']
            folder: Optional[str]=      None,
    ):
        """ prepares 3D plot of the Cloud VPoints """

        columns = sorted(list(self._vpointsD[0].point.keys()))[:3] if not axes else [] + axes

        if len(columns) < 2:
            self.logger.warning('Cannot prepare 3D plot for less than two axes')

        else:

            valLL = [[sp.point[key] for key in columns] for sp in self._vpointsD.values()]

            # eventually add score
            if len(columns) < 4:
                columns += ['value']
                valLL = [vl + [sp.value] for vl,sp in zip(valLL, self._vpointsD.values())]
            three_dim(
                xyz=        valLL,
                name=       name,
                x_name=     columns[0],
                y_name=     columns[1],
                z_name=     columns[2],
                val_name=   'val',
                save_FD=    folder)

    @property
    def vpoints(self) -> List[VPoint]:
        return list(self._vpointsD.values())

    def __len__(self):
        """ number of VPoints in the Cloud """
        return len(self._vpointsD)

    def __str__(self):

        vpoints = self.vpoints
        vpoints.sort(key=lambda x: x.value, reverse=True)
        keys = sorted(list(vpoints[0].point.keys()))
        kw = {k: len(k) for k in keys}
        kw['_name'] = 0
        kw['_value'] = 1

        print_vals = []
        for vp in vpoints:

            pv = {}
            print_vals.append(pv)
            for k in keys:

                val = vp.point[k]

                val_str = ''
                if type(val) is float:
                    val_str = float_to_str(val, fill=False)
                if type(val) is bool:
                    val_str = f'{str(val):5}'
                if not val_str:
                    val_str = str(val)

                pv[k] = val_str

                lv = len(val_str)
                if lv > kw[k]:
                    kw[k] = lv

            ln = len(vp.name) if vp.name else 0
            if ln > kw['_name']:
                kw['_name'] = ln

            lv = len(float_to_str(vp.value, fill=False))
            if lv > kw['_value']:
                kw['_value'] = lv

        s = ' ' * (kw['_name'] + 5 + kw['_value'] + 2)
        s += ' '.join([f'{k:{kw[k]}}' for k in keys]) + '\n'
        for vp,pv in zip(vpoints,print_vals):
            val = f'{float_to_str(vp.value, fill=False):{kw["_value"]}}' if vp.value is not None else '-' * kw['_value']
            s += f'{vp.name:{kw["_name"]}} val:{val}  '
            s += ' '.join([f'{pv[k]:{kw[k]}}' for k in keys]) + '\n'

        return s[:-1]