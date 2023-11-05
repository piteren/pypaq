import math
from typing import Sized, List, Tuple, Optional, Dict, Union

from pypaq.lipytools.pylogger import get_pylogger
from pypaq.lipytools.stats import mam
from pypaq.lipytools.plots import three_dim
from pypaq.pms.base import POINT, point_str
from pypaq.pms.paspa import PaSpa


class VPoint:
    """ Valued Point
    with unique id & float value """

    def __init__(
            self,
            point: POINT,
            id: Optional[int] =     None,
            value: Optional[float]= None,
    ):
        self.point = point
        self.id = id
        self.value = value

    def __str__(self):
        id_nfo = f'#{self.id:4}' if self.id is not None else '_'
        return f'{id_nfo} [val: {self.value:.8f}] {point_str(self.point)}'


class PointsCloud(Sized):
    """ cloud of Valued Points """

    def __init__(
            self,
            paspa: PaSpa,   # space of this PointsCloud
            logger=     None,
            loglevel=   20):

        if not logger:
            logger = get_pylogger(level=loglevel)
        self.logger = logger
        self.logger.info('*** PointsCloud *** initializing..')

        self.paspa = paspa

        # those below are updated with each call to update_cloud()
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