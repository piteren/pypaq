from abc import ABC, abstractmethod
import numpy as np
from sklearn.svm import SVR
from typing import List, Optional, Tuple, Dict

from pypaq.pytypes import NPL, NUM
from pypaq.lipytools.pylogger import get_pylogger
from pypaq.pms.base import PMSException
from pypaq.pms.paspa import PaSpa
from pypaq.pms.points_cloud import VPoint


class SpaceEstimator(ABC):

    def __init__(
            self,
            logger=         None,
            loglevel=       20):
        if not logger:
            logger = get_pylogger(level=loglevel)
        self.logger = logger

    # extracts X & y from vpoints & space
    @staticmethod
    def _extract_Xy(vpoints:List[VPoint], space:PaSpa) -> Tuple[np.ndarray, np.ndarray]:

        points = [vp.point for vp in vpoints]
        points_normalized = [space.point_normalized(p) for p in points]
        keys = sorted(list(points_normalized[0].keys()))
        points_feats = [[pn[k] for k in keys] for pn in points_normalized]

        scores = [vp.value for vp in vpoints]

        return np.asarray(points_feats), np.asarray(scores)

    # updates model with given data, returns dict with update info
    @abstractmethod
    def update(self, X_new:NPL, y_new:NPL) -> Dict[str,NUM]:
        pass

    # updates model with given VPoints, returns dict with update info
    def update_vpoints(self, vpoints:List[VPoint], space:PaSpa) -> Dict[str,NUM]:
        X, y = SpaceEstimator._extract_Xy(vpoints, space)
        return self.update(X, y)

    # predicts
    @abstractmethod
    def predict(self, X:NPL) -> np.ndarray:
        pass

    def predict_vpoints(self, vpoints:List[VPoint], space:PaSpa) -> np.ndarray:
        X, y = SpaceEstimator._extract_Xy(vpoints, space)
        return self.predict(X)

    # MSE average weighted loss
    @staticmethod
    def loss(
            model,
            y_test: NPL,
            X_test: Optional[NPL]=  None,
            preds: Optional[NPL]=   None,
            weights: Optional[NPL]= None,
    ) -> float:

        if X_test is None and preds is None:
            raise PMSException('\'X_test\' or \'preds\' must be given')

        if preds is None:
            preds = model.predict(X=X_test)

        if type(preds) is not np.ndarray:
            preds = np.asarray(preds)
        if type(y_test) is not np.ndarray:
            y_test = np.asarray(y_test)

        l = (preds - y_test) ** 2

        if weights is not None:
            if type(weights) is not np.ndarray:
                weights = np.asarray(weights)
            l *= weights

        return float(np.mean(l))

    # Estimator status
    @property
    def fitted(self) -> bool:
        return False

    # Estimator state
    @property
    def state(self) -> Dict:
        return {}

    @classmethod
    @abstractmethod
    def from_state(cls, state:Dict, logger):
        pass

    def __str__(self):
        return 'SpaceEstimator'


# Support Vector Regression (SVR) with Radial Basis Function (RBF) kernel based Space Estimator
class RBFRegressor(SpaceEstimator):

    # C & gamma parameters possible values
    VAL = {
        'c': [100, 10, 1, 0.1, 0.01],
        'g': [0.01, 0.1, 1, 10, 100]}

    def __init__(
            self,
            epsilon: float=     0.01,
            num_tests: int=     9,      # number of cross-validation tests while updating
            test_size: float=   0.25,   # size of test data (factor of all)
            y_cut_off: float=   0.1,    # value of y with lower loss (for sparse y)
            seed=               123,
            **kwargs,
    ):

        super().__init__(**kwargs)

        self._indexes =  {'c':0, 'g':0}
        self._epsilon = epsilon

        self._model = None

        self.data_X: Optional[NPL] = None
        self.data_y: Optional[NPL] = None

        self._num_tests = num_tests
        self._test_size = test_size
        self._y_cut_off = y_cut_off

        np.random.seed(seed)

    # builds SVR RBF model from given indexes of parameters
    def _build_model(self, cix:Optional[int]=None, gix:Optional[int]=None) -> SVR:
        if cix is None: cix = self._indexes['c']
        if gix is None: gix = self._indexes['g']
        return SVR(
            kernel=     "rbf",                      # Radial Basis Function kernel
            C=          RBFRegressor.VAL['c'][cix], # regularization, controls error with margin, lower C -> larger margin, more support vectors, longer fitting time
            gamma=      RBFRegressor.VAL['g'][gix], # controls shape of decision boundary, larger value for more complex
            epsilon=    self._epsilon)              # threshold for what is considered an acceptable error rate in the training data

    # fits model, returns test loss
    @staticmethod
    def _fit(
            model,
            X_train: NPL,
            y_train: NPL,
            X_test: NPL,
            y_test: NPL,
            weights: Optional[NPL]= None,
    ) -> float:
        model.fit(X=X_train, y=y_train)
        return RBFRegressor.loss(model=model, X_test=X_test, y_test=y_test, weights=weights)

    # prepares weights for y
    def _weights(self, y:NPL) -> np.ndarray:
        y_min = np.min(y)
        y_max = np.max(y)
        y_cut = y_min + (y_max - y_min) * self._y_cut_off
        factor = 1 - np.mean((y <= y_cut).astype(int))
        factor = max(0.1, factor)
        return np.where(y > y_cut, 1, factor)

    def update(self, X_new:NPL, y_new:NPL) -> Dict[str,NUM]:
        """ concatenates data, tries to update model params, fits & returns some info """

        self.logger.debug(f'preparing for update with {self._num_tests}X cross-validation')

        loss_on_new = 0
        if self.fitted:
            loss_on_new = self.loss(
                model=  self._model,
                X_test= X_new,
                y_test= y_new)

        # add new data
        self.data_X = np.concatenate([self.data_X, X_new]) if self.data_X is not None else X_new
        self.data_y = np.concatenate([self.data_y, y_new]) if self.data_y is not None else y_new

        ### prepare valid future configs

        m_configs = {
            'current':  {'cix': self._indexes['c'],     'gix': self._indexes['g']},     # current
            'cH':       {'cix': self._indexes['c'] + 1, 'gix': self._indexes['g']},     # c higher
            'cL':       {'cix': self._indexes['c'] - 1, 'gix': self._indexes['g']},     # c lower
            'gH':       {'cix': self._indexes['c'],     'gix': self._indexes['g'] + 1}, # g higher
            'gL':       {'cix': self._indexes['c'],     'gix': self._indexes['g'] - 1}} # g lower

        not_valid = []
        for k in m_configs:
            for p in m_configs[k]:
                if m_configs[k][p] in [-1,len(RBFRegressor.VAL[p[0]])]:
                    not_valid.append(k)
        not_valid = list(set(not_valid))
        for k in not_valid:
            m_configs.pop(k)
        self.logger.debug(f'valid alternative models configs for update: {m_configs}')

        models = {config: self._build_model(**m_configs[config]) for config in m_configs}

        acc_loss = {k: 0 for k in m_configs}
        for _ in range(self._num_tests):

            ### prepare data split

            data_size = len(self.data_X)
            train_size = int(data_size * (1-self._test_size))
            choice = np.random.choice(range(data_size), size=train_size, replace=False)
            tr_sel = np.zeros(data_size, dtype=bool)
            tr_sel[choice] = True
            ts_sel = ~tr_sel

            X_train = self.data_X[tr_sel]
            y_train = self.data_y[tr_sel]
            X_test = self.data_X[ts_sel]
            y_test = self.data_y[ts_sel]

            losses = {config: RBFRegressor._fit(
                model=      models[config],
                X_train=    X_train,
                y_train=    y_train,
                X_test=     X_test,
                y_test=     y_test,
                weights=    self._weights(y=y_test)) for config in models}

            for k in losses:
                acc_loss[k] += losses[k]

        for k in acc_loss:
            acc_loss[k] /= self._num_tests
        self.logger.debug(f'loss achieved with each config: {acc_loss}')

        acc_loss_current = acc_loss.pop('current')

        # search for k to update, priority for lower
        update_k = None
        losses_sorted = sorted([(k,l) for k,l in acc_loss.items()], key=lambda x:x[1])
        for e in losses_sorted:
            if e[1] < acc_loss_current:
                update_k = e[0]
                break
        self.logger.debug(f'RBFRegressor is updating: {update_k}')

        if update_k is not None:
            self._indexes[update_k[0]] += 1 if update_k[1] == 'H' else -1
            self._model = models[update_k]
        else:
            self._model = models['current']

        self.logger.debug(f' > current config of indexes: {self._indexes}')

        # finally fit with all data
        loss_all_data_weighted = RBFRegressor._fit(
            model=      self._model,
            X_train=    self.data_X,
            y_train=    self.data_y,
            X_test=     self.data_X,
            y_test=     self.data_y,
            weights=    self._weights(y=self.data_y))

        return {
            'loss_on_new':              loss_on_new,
            'loss_all_data_weighted':   loss_all_data_weighted,
            'c_ix':                     self._indexes['c'],
            'g_ix':                     self._indexes['g']}

    def predict(self, x:NPL) -> np.ndarray:
        if not self.fitted:
            msg = 'RBFRegressor needs to be fitted before predict'
            self.logger.error(msg)
            raise Exception(msg)
        return self._model.predict(X=x)

    @property
    def fitted(self) -> bool:
        return self._model is not None

    # returns object state
    @property
    def state(self) -> Dict:
        return {
            'indexes':      self._indexes,
            'epsilon':      self._epsilon,
            'num_tests':    self._num_tests,
            'test_size':    self._test_size,
            'y_cut_off':    self._y_cut_off}

    # builds object from a state
    @classmethod
    def from_state(cls, state:Dict, logger):
        reg = cls(
            epsilon=    state['epsilon'],
            num_tests=  state['num_tests'],
            test_size=  state['test_size'],
            y_cut_off=  state['y_cut_off'],
            logger=     logger)
        reg._indexes = state['indexes']
        logger.info(f'RBFRegressor build from state: {state}')
        return reg

    def __str__(self):
        c = RBFRegressor.VAL['c'][self._indexes['c']]
        g = RBFRegressor.VAL['g'][self._indexes['g']]
        return f'SVR RBF, C:{c}, gamma:{g}, data:{len(self.data_X)}'