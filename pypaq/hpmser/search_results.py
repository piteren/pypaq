import random
from typing import Sized, List, Tuple, Optional, Iterable
import warnings

from pypaq.hpmser.helpers import _str_weights
from pypaq.lipytools.files import r_pickle, w_pickle
from pypaq.lipytools.plots import three_dim
from pypaq.pms.paspa import PaSpa
from pypaq.pms.base import POINT, point_str


# Search Results List [SeRes] with some methods
class SRL(Sized):

    # single Search Result
    class SeRes:

        def __init__(
                self,
                point: POINT,
                score=      None):
            self.id = None              # to be updated, id of SeRes, set by SRL: id = len(SRL)
            self.point = point
            self.score = score
            self.smooth_score = None    # to be updated

        def __str__(self): return f'SeRes: id:{self.id}, point:{self.point}, score:{self.score}, smooth_score{self.smooth_score}'

    def __init__(
            self,
            logger,
            paspa: Optional[PaSpa]=     None,   # parameters space of this SRL
            name: str=                  'SRL',
            np_smooth: int=             3,      # (NPS) Number of Points taken into account while calculating smooth score - default / starting value
            plot_axes: list=            None):  # list with axes names (max 3), eg: ['drop_a','drop_b','loss']

        self.name = name
        self.logger = logger
        self.logger.info(f'*** SRL : {self.name} *** initializing..')

        self.paspa = paspa

        self.__np_smooth = np_smooth        # current np_smooth of SeRes
        self.plot_axes = plot_axes


        self.__srL: List[SRL.SeRes] = []    # sorted periodically by smooth_score
        self.__smoothed_and_sorted = True   # flag (state) - keeps info about SRL that is smoothed & sorted
        self.__distances = []               # distances cache (by SeRes id)
        self.__scores = []                  # scores cache (by SeRes id)
        self.__avg_dst = 1                  # average distance of SRL for self.__np_smooth
        self.prec = 8                       # print precision, will be updated while adding new points

    # ****************************************************************************************************** load & save

    def __get_srl_path(self, save_dir: str) -> str: return f'{save_dir}/{self.name}.srl'

    def __get_srl_backup_path(self, save_dir: str) -> str: return f'{save_dir}/{self.name}.srl.backup'

    # loads (alternatively from backup)
    def load(self, save_dir :str):

        self.logger.info(f' > SRL {self.name} loading form {save_dir}..')

        try:
            obj = r_pickle(self.__get_srl_path(save_dir))
        except Exception as e:
            self.logger.warning(f' SRL {self.name} got exception: {str(e)} while loading, using backup file')
            obj = r_pickle(self.__get_srl_backup_path(save_dir))

        self.paspa =                    obj.paspa
        self.__np_smooth =              obj.__np_smooth
        self.plot_axes =                obj.plot_axes

        self.__srL =                    obj.__srL
        self.__smoothed_and_sorted =    obj.__smoothed_and_sorted
        self.__distances =              obj.__distances
        self.__scores =                 obj.__scores
        self.__avg_dst =                obj.__avg_dst
        self.prec=                      obj.prec

        self.logger.info(f' > SRL loaded {len(self.__srL)} results')

    # saves with backup
    def save(self, folder :str):

        # backup copy previous
        old_res = r_pickle(self.__get_srl_path(folder))
        if old_res: w_pickle(old_res, self.__get_srl_backup_path(folder))

        w_pickle(self, self.__get_srl_path(folder))
        self.plot(folder=folder)

    # ************************************************************************************************ getters & setters

    # returns top SeRes (max smooth_score)
    def get_top_SR(self) -> SeRes or None:
        if self.__srL:
            if not self.__smoothed_and_sorted:
                warnings.warn('SRL asked to return top SR while SRL is not smoothed_and_sorted - have to sort!')
                self.smooth_and_sort()
            return self.__srL[0]
        return None

    def get_SR(self, id: int) -> SeRes or None:
        for sr in self.__srL:
            if sr.id == id: return sr
        return None

    # returns distance between two points, if points are SeRes type uses cached distance
    def get_distance(self,
            pa: POINT or SeRes,
            pb: POINT or SeRes) -> float:
        if type(pa) is SRL.SeRes and type(pb) is SRL.SeRes: return self.__distances[pa.id][pb.id]
        if type(pa) is SRL.SeRes: pa = pa.point
        if type(pb) is SRL.SeRes: pb = pb.point
        return self.paspa.distance(pa, pb)

    def get_avg_dst(self): return self.__avg_dst

    # max of min distances of SRL: max(min_distance)
    def get_mom_dst(self): return max([min(d[1:]) if d[1:] else 0 for d in self.__distances])

    # returns sample with policy and estimated score
    def get_opt_sample(self,
            prob_opt,   # probability of optimized sample
            n_opt,      # number of optimized samples
            prob_top,   # probability of sample from area of top
            n_top,      # number of top samples
            avg_dst     # distance for sample from area of top
    ) -> Tuple[POINT,float]:

        prob_rnd = 1 - prob_opt - prob_top
        if random.random() < prob_rnd or len(self.__srL) < 10: sample = self.paspa.sample_point_GX() # one random point
        else:
            if random.random() < prob_opt/(prob_top+prob_opt): points = [self.paspa.sample_point_GX() for _ in range(n_opt + 1)] # some random points ...last for reference
            # top points
            else:
                n_top += 1 # last for reference
                if n_top > len(self.__srL): n_top = len(self.__srL)
                points = [self.paspa.sample_point_GX(
                    point_main=     self.__srL[ix].point,
                    noise_scale=    avg_dst) for ix in range(n_top)] # top points

            scores = [self.smooth_point(p)[0] for p in points]

            all_pw = list(zip(points, scores))
            all_pw.sort(key=lambda x: x[1], reverse=True)
            maxs = all_pw[0][1]
            subs = all_pw.pop(-1)[1]
            mins = all_pw[-1][1]

            all_p, all_w = zip(*all_pw)
            all_w = [w - subs for w in all_w]
            all_p = list(all_p)
            sample = random.choices(all_p, weights=all_w, k=1)[0]
            pf = f'.{self.prec}f'
            print(f'   % sampled #{all_p.index(sample)}/{len(all_p)} from: {maxs:{pf}}-{mins:{pf}} {_str_weights(all_w, float_prec=self.prec)}')

        est_score, _, _ =  self.smooth_point(sample)

        return sample, est_score

    # returns sample chosen with policy and its estimated score
    def get_opt_sample_GX(
            self,
            prob_opt,   # probability of optimized sample
            n_opt,      # number of optimized samples
            prob_top,   # probability of sample from area of top
            n_top,      # number of top samples
            avg_dst     # distance for sample from area of top
    ) -> Tuple[POINT,float]:

        if random.random() < 1-prob_opt-prob_top or len(self.__srL) < 10:
            sample = self.paspa.sample_point_GX() # one random point
        else:
            # choice from better half of random points
            if random.random() < prob_opt/(prob_top+prob_opt):
                points = [self.paspa.sample_point_GX() for _ in range(2 * n_opt)]   # 2*some random points
                scores = [self.smooth_point(p)[0] for p in points]                  # their scores
                all_pw = list(zip(points, scores))
                all_pw.sort(key=lambda x: x[1], reverse=True)                       # sorted
                all_pw = all_pw[:len(all_pw)//2]                                    # take better half
                maxs = all_pw[0][1]
                subs = all_pw.pop(-1)[1]
                mins = all_pw[-1][1]
                all_p, all_w = zip(*all_pw)
                all_w = [w - subs for w in all_w]
                all_p = list(all_p)
                sample = random.choices(all_p, weights=all_w, k=1)[0]
                pf = f'.{self.prec}f'
                self.logger.info(f'   % sampled #{all_p.index(sample)}/{len(all_p)} from: {maxs:{pf}}-{mins:{pf}} {_str_weights(all_w, float_prec=self.prec)}')
            # GX from top points
            else:
                n_top += 1 # last for reference
                if n_top > len(self.__srL): n_top = len(self.__srL)
                top_sr = self.__srL[:n_top]                             # top SR
                scores = [p.smooth_score for p in top_sr]               # their scores
                mins = min(scores)
                scores = [s-mins for s in scores]                       # reduced scores
                sra, srb = random.choices(top_sr, weights=scores, k=2)  # two SR
                sample = self.paspa.sample_point_GX(
                    point_main=                 sra.point,
                    point_scnd=                 srb.point,
                    noise_scale=                avg_dst)
                pf = f'.{self.prec}f'
                self.logger.info(f'   % sampled GX from: {sra.smooth_score:{pf}} and {srb.smooth_score:{pf}}')
        est_score, _, _ =  self.smooth_point(sample)
        return sample, est_score

    # returns n closest SeRes to given point
    def __get_n_closest(
            self,
            point: POINT or SeRes,
            n: Optional[int]=   None) -> List[SeRes]:
        if not n: n = self.__np_smooth
        if len(self.__srL) <= n:
            return [] + self.__srL
        else:
            id_dst = \
                list(zip(range(len(self.__srL)), self.__distances[point.id])) \
                    if type(point) is SRL.SeRes else \
                    [(sr.id, self.get_distance(point, sr.point)) for sr in self.__srL]
            id_dst.sort(key=lambda x: x[1]) # sort by distance to this point
            return [self.get_SR(id[0]) for id in id_dst[:n]]

    # sets np_smooth, then smooths and sorts
    def set_np_smooth(self, np_smooth: int):
        if np_smooth != self.__np_smooth:
            self.__np_smooth = np_smooth
            self.smooth_and_sort()

    # adds new result, caches distances, smooths & sorts
    def add_result(self,
            point: POINT,
            score: float,
            force_no_update=    False # aborts calculation of smooth score and sorting
    ) -> SeRes:

        sr = SRL.SeRes(point,score)
        sr.id = len(self.__srL)

        # update cached distances
        sr_dist = []
        id_point = [(s.id,s.point) for s in self.__srL]
        id_point.sort(key= lambda x:x[0]) # sort by id
        for id,point in id_point:
            d = self.paspa.distance(sr.point,point)
            sr_dist.append(d)
            self.__distances[id].append(d)
        sr_dist.append(0)

        self.__distances.append(sr_dist)

        self.__scores.append(score) # update cached score

        self.__srL.append(sr) # add SeRes

        if force_no_update:
            sr.smooth_score = sr.score
            self.__smoothed_and_sorted = False
        else: self.smooth_and_sort()

        if score > 0.01: self.prec = 4

        return sr

    # returns (smooth_score, average_distance, all scores sorted by distance) for given point or SeRes
    def smooth_point(self,
            point: POINT or SeRes,
            np_smooth: Optional[int]=   None # used if given, otherwise self.__np_smooth
    ) -> Tuple[float, float, List[float]]:

        # case: no points in srL
        smooth_score_np = 0     # smooth score for self.__np_smooth
        avg_dst_np =      1     # average distance for np_smooth
        all_scores =     [0]    # np scores sorted by distance

        if not np_smooth: np_smooth = self.__np_smooth
        if self.__srL:
            score_dst = \
                list(zip(self.__scores, self.__distances[point.id])) \
                    if type(point) is SRL.SeRes else \
                    [(sr.score, self.get_distance(point, sr.point)) for sr in self.__srL]
            score_dst.sort(key=lambda x: x[1])  # sort by distance to this point
            score_dst_np = score_dst[:np_smooth + 1] # trim to np_smooth points (+1 point for reference)

            # case of one/two points in score_dst_np
            if len(score_dst_np) < 3:
                smooth_score_np = score_dst_np[0][0]  # closest point score
                all_scores = [score_dst_np[0][0]]
            else:
                all_scores, all_dst = zip(*score_dst_np)  # scores, distances

                max_dst = all_dst[-1]  # distance of last(reference) point

                # remove last (reference)
                all_dst = all_dst[:-1]
                all_scores = all_scores[:-1]

                # set weights for scores
                weights = []
                if max_dst: weights = [(max_dst-d)/max_dst for d in all_dst]  # try with distance based weights <1;0>
                if sum(weights) == 0: weights = [1] * len(all_dst)  # naive baseline / equal for case: (max_dst == 0) or (max_dst-d == 0)

                wall_scores = [all_scores[ix]*weights[ix] for ix in range(len(all_scores))]
                smooth_score_np = sum(wall_scores) / sum(weights) # weighted score
                avg_dst_np = sum(all_dst) / len(all_dst)

        return smooth_score_np, avg_dst_np, all_scores

    # smooths self.__srL and sorts by SeRes.smooth_score
    def smooth_and_sort(self):

        if self.__srL:
            avg_dst = []
            for sr in self.__srL:
                sr.smooth_score, ad, _ = self.smooth_point(sr)
                avg_dst.append(ad)
            self.__avg_dst = sum(avg_dst)/len(avg_dst)

            self.__srL.sort(key=lambda x: x.smooth_score, reverse=True)

        self.__smoothed_and_sorted = True

    def log_distances(self):
        for dl in self.__distances:
            s = ''
            for d in dl: s += f'{d:.2f} '
            self.logger.info(s)

    def nice_str(
            self,
            n_top=                  20,
            top_nps: Iterable[int]= (3,5,9),
            all_nps: Optional[int]= 3):

        pf = f'.{self.prec}f'

        re_str = ''
        if all_nps: re_str += f'Search run {self.name}, {len(self.__srL)} results:\n\n{self.paspa}\n\n'

        if len(self.__srL) < n_top: n_top = len(self.__srL)
        orig_nps = self.__np_smooth
        top_nps = list(top_nps)
        for nps in top_nps:
            self.set_np_smooth(nps)

            re_str += f'TOP{n_top} results for NPS {nps} (avg_dst:{self.__avg_dst:.3f}):'
            if top_nps.index(nps) == 0: re_str += ' -- id smooth [local] [max-min] avg_dst {params..}\n'
            else: re_str += '\n'

            for srIX in range(n_top):
                sr = self.__srL[srIX]
                ss_np, avg_dst, all_scores = self.smooth_point(sr)
                re_str += f'{sr.id:4d} {ss_np:{pf}} [{sr.score:{pf}}] [{max(all_scores):{pf}}-{min(all_scores):{pf}}] {avg_dst:.3f} {point_str(sr.point)}\n'

        self.set_np_smooth(orig_nps)
        top_sr_nps = self.get_top_SR()
        n_closest = self.__get_n_closest(top_sr_nps, n=self.__np_smooth)
        re_str += f'{self.__np_smooth} closest points to TOP (NPS {self.__np_smooth}):\n'
        for sr in n_closest: re_str += f'{sr.id:4d} [{sr.score:{pf}}] {point_str(sr.point)}\n'

        if all_nps and len(self.__srL) > n_top:
            self.set_np_smooth(all_nps)
            re_str += f'\nALL results for NPS {all_nps} (avg_dst:{self.__avg_dst:.3f}):\n'
            for sr in self.__srL:
                ss_np, avg_dst, all_scores = self.smooth_point(sr)
                re_str += f'{sr.id:4d} {ss_np:{pf}} [{sr.score:{pf}}] [{max(all_scores):{pf}}-{min(all_scores):{pf}}] {avg_dst:.3f} {point_str(sr.point)}\n'

        self.set_np_smooth(orig_nps)
        return re_str

    # 3D plot
    def plot(
            self,
            smooth_score=   True,   # for True color_"axis" == smooth_score, else score
            folder: str=    None):

        columns = sorted(list(self.__srL[0].point.keys()))[:3] if not self.plot_axes else [] + self.plot_axes
        valLL = [[res.point[key] for key in columns] for res in self.__srL]

        # eventually add score
        if len(columns) < 3:
            valLL = [vl + [res.score] for vl,res in zip(valLL,self.__srL)]
            columns += ['score']

        # eventually add smooth_score (for one real axis)
        if len(columns) < 3:
            valLL = [vl + [res.smooth_score] for vl,res in zip(valLL,self.__srL)]
            columns += ['smooth_score']

        # add color "axis" data
        columns += ['smooth_score' if smooth_score else 'score']
        valLL = [valL + [res.smooth_score if smooth_score else res.score] for valL, res in zip(valLL, self.__srL)]

        three_dim(
            xyz=        valLL,
            name=       self.name,
            x_name=     columns[0],
            y_name=     columns[1],
            z_name=     columns[2],
            val_name=   columns[3],
            save_FD=    folder)

    def __len__(self): return len(self.__srL)