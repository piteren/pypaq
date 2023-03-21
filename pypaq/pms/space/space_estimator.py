import numpy as np
from sklearn.svm import SVR
from typing import List, Union, Optional, Tuple

from pypaq.pms.space.space_point import SPoint
from pypaq.pms.space.paspa import PaSpa

NPL = Union[List, np.ndarray]


# MSE average loss for Estimators
def loss(model, X_new:NPL, y_new:NPL) -> float:
    preds = model.predict(X=X_new)
    return sum([(a - b) ** 2 for a, b in zip(preds, y_new)]) / len(y_new)


class SpaceEstimator:

    # extracts X & y from spoints ands space
    @staticmethod
    def _extract_Xy(spoints:List[SPoint], space:PaSpa) -> Tuple[np.ndarray, np.ndarray]:

        points = [sp.point for sp in spoints]
        points_normalized = [space.point_normalized(p) for p in points]
        keys = sorted(list(points_normalized[0].keys()))
        points_feats = [[pn[k] for k in keys] for pn in points_normalized]

        scores = [sp.score for sp in spoints]

        return np.asarray(points_feats), np.asarray(scores)

    # updates model with given data, returns loss of new model for given data
    def update(self, X_new:NPL, y_new:NPL) -> float:
        pass


    def update_spoints(self, spoints:List[SPoint], space:PaSpa) -> float:
        X, y = SpaceEstimator._extract_Xy(spoints, space)
        return self.update(X, y)

    # predicts
    def predict(self, X:NPL) -> np.ndarray:
        pass


    def predict_spoints(self, spoints:List[SPoint], space:PaSpa) -> np.ndarray:
        X, y = SpaceEstimator._extract_Xy(spoints, space)
        return self.predict(X)


# SVR RBF based Space Estimator
class RBFRegressor(SpaceEstimator):

    # C & gamma parameters possible values
    VAL = {
        'c': [100, 10, 1, 0.1, 0.01],
        'g': [0.01, 0.1, 1, 10, 100]}

    def __init__(
            self,
            epsilon: float= 0.01,
            num_tries: int= 2,  # how many times param with next ix needs to improve to increase ix
    ):

        self._epsilon = epsilon
        self._num_tries = num_tries
        self._indexes = {'c':0, 'g':0}
        self._improved = {'c':0, 'g':0}

        self._model = self._build_model()

        self.data_X: Optional[NPL] = None
        self.data_y: Optional[NPL] = None
        self._fitted: bool = False

    # builds SVR RBF model from given indexes of parameters
    def _build_model(self, cix:Optional[int]=None, gix:Optional[int]=None) -> SVR:
        if cix is None: cix = self._indexes['c']
        if gix is None: gix = self._indexes['g']
        return SVR(
            kernel=     "rbf",
            C=          RBFRegressor.VAL['c'][cix], # regularization, controls error with margin, lower C -> larger margin, more support vectors, longer fitting time
            gamma=      RBFRegressor.VAL['g'][gix], # controls shape of decision boundary, larger value for more complex
            epsilon=    self._epsilon)              # threshold for what is considered an acceptable error rate in the training data

    # adds given data, then fits model
    def _fit(self, X_new:Optional[NPL]=None, y_new:Optional[NPL]=None) -> None:

        if X_new is not None:

            # convert if needed
            if type(X_new) is not np.ndarray: X_new = np.asarray(X_new)
            if type(y_new) is not np.ndarray: y_new = np.asarray(y_new)

            # append
            self.data_X = np.concatenate([self.data_X, X_new]) if self.data_X is not None else X_new
            self.data_y = np.concatenate([self.data_y, y_new]) if self.data_y is not None else y_new

        self._model.fit(X=self.data_X, y=self.data_y)
        self._fitted = True

    # tries to update model params, adds data, then fits, returns loss of new model for given data
    def update(self, X_new:NPL, y_new:NPL) -> float:

        if self._fitted:

            loss_current = loss(model=self._model, X_new=X_new, y_new=y_new)

            new_indexes = {
                'c': self._indexes['c'] + 1,
                'g': self._indexes['g'] + 1}

            loss_updated = {}
            for k in new_indexes:

                # no possible update for k
                if new_indexes[k] == len(RBFRegressor.VAL[k]):
                    loss_updated[k] = loss_current

                # check model with k updated
                else:
                    params = {f'{p}ix': self._indexes[p] for p in self._indexes}
                    params[f'{k}ix'] = new_indexes[k]
                    model = self._build_model(**params)
                    model.fit(X=self.data_X, y=self.data_y)
                    loss_updated[k] = loss(model=model, X_new=X_new, y_new=y_new)

            # increase OR reset
            for k in loss_updated:
                if loss_updated[k] < loss_current: self._improved[k] += 1
                else:                              self._improved[k] = 0

            # search for k to update, priority for lower
            update_k = None
            loss_updated_sorted = sorted([(k,l) for k,l in loss_updated.items()], key=lambda x:x[1])
            for e in loss_updated_sorted:
                k = e[0]
                if self._improved[k] == self._num_tries:
                    update_k = k
                    break

            # update & build new model, reset counters
            if update_k is not None:
                self._indexes[update_k] += 1
                self._model = self._build_model()
                self._improved = {k:0 for k in self._improved} # reset

        # fit model on concatenated data
        self.data_X = np.concatenate([self.data_X, X_new]) if self.data_X is not None else X_new
        self.data_y = np.concatenate([self.data_y, y_new]) if self.data_y is not None else y_new

        self._fit()

        return loss(model=self._model, X_new=X_new, y_new=y_new)


    def predict(self, x:NPL) -> np.ndarray:
        if not self._fitted:
            raise Exception('RBFRegressor needs to be fitted before predict')
        return self._model.predict(X=x)


    def __str__(self):
        c = RBFRegressor.VAL['c'][self._indexes['c']]
        g = RBFRegressor.VAL['g'][self._indexes['g']]
        return f'SVR RBF, C:{c}, gamma:{g}, data:{len(self.data_X)}'