import numpy as np
import pandas as pd
from copy import deepcopy
from inspect import getfullargspec

from psc.groups import MISSING_VALUE


class ClassificationModel(object):

    def __init__(self, predict_handle, proba_handle, group_encoder=None, **kwargs):
        self._predict_handle = deepcopy(predict_handle)
        self._proba_handle = deepcopy(proba_handle)
        self._model_type = deepcopy(kwargs.get('model_type'))
        self._model_info = deepcopy(kwargs.get('model_info'))
        self._training_info = deepcopy(kwargs.get('training_info'))
        self._training_groups = self._training_info.get('training_groups', [])

        # group encoder
        self._group_encoder = group_encoder
        self._supported_groups = self._group_encoder.groups if group_encoder is not None else set([])
        self._required_group_attributes = set(self._group_encoder.names) if group_encoder is not None else set([])

    @property
    def training_groups(self):
        return self._training_groups

    @property
    def model_type(self):
        return self._model_type

    @property
    def model_info(self):
        return self._model_info

    @property
    def supported_groups(self):
        """tuple containing the names of the groups that can be assigned to this classifier"""
        return self._supported_groups

    @property
    def required_group_attributes(self):
        """tuple containing the group attributes that are required to predict with this classifier"""
        return self._required_group_attributes

    def predict(self, X, Gr=None):
        pass

    def predict_proba(self, X, Gr=None):
        pass

    def predict_for(self, X, G, group_label, use_missing_val=False):
        pass

    def predict_proba_for(self, X, G, group_label, use_missing_val=False):
        pass

    def __repr__(self):
        return f'ClassificationModel<method: {self.model_type}, supported_groups={self.supported_groups}, ' \
               f'group_attributes={self.required_group_attributes}> '


class LinearClassificationModel(ClassificationModel):
    LINEAR_MODEL_TYPE = 'linear'
    SUPPORTED_MODEL_TYPES = [LINEAR_MODEL_TYPE]

    def __init__(self, predict_handle, proba_handle, group_encoder=None, **kwargs):
        super().__init__(predict_handle, proba_handle, group_encoder, **kwargs)
        assert self._model_type in LinearClassificationModel.SUPPORTED_MODEL_TYPES, "unsupported model type"
        assert np.isfinite(self._model_info['intercept'])
        self._intercept = float(self._model_info['intercept'])
        assert np.isfinite(self._model_info['coefficients']).all()
        self._coefficients = np.array(self._model_info['coefficients']).flatten()
        self._impute_group_attributes = self._model_info.get('impute_group_attributes', False)

        if self._predict_handle is None:
            coefficient_idx = self._model_info['coefficient_idx']
            self._predict_handle = lambda X: np.sign(np.array(
                X[:, coefficient_idx].dot(self._coefficients) + self._intercept))  # > 0.5
        assert callable(self._predict_handle)
        spec_predict = getfullargspec(self._predict_handle)
        assert 'X' in spec_predict.args

        if self._proba_handle is None:
            coefficient_idx = self._model_info['coefficient_idx']
            self._proba_handle = lambda X: 1. / (
                    1. + np.exp(-(X[:, coefficient_idx].dot(self._coefficients) + self._intercept)))
        assert callable(self._proba_handle)
        spec_proba = getfullargspec(self._predict_handle)
        assert 'X' in spec_proba.args

    @property
    def intercept(self):
        return self._intercept

    @property
    def coefficients(self):
        return self._coefficients

    def predict(self, X, G=None):
        if self._impute_group_attributes:
            G = pd.DataFrame(np.repeat(self._group_encoder.mode.values, X.shape[0], axis=0),
                             columns=self._group_encoder.mode.columns)
        if G is not None and not G.empty:
            assert self._group_encoder is not None
            Z = self._group_encoder.to_dummies(df=G)
            if X is not None:
                X = np.hstack([Z, X])
            else:
                X = Z
        return self._predict_handle(X).flatten()

    def predict_proba(self, X, G=None):
        if G is not None and not G.empty:
            Z = self._group_encoder.to_dummies(df=G)
            if X is not None:
                X = np.hstack([Z, X])
            else:
                X = Z
        probs = self._proba_handle(X)
        if len(probs.shape) > 1:
            probs = probs[:, 1]
        return probs.flatten()

    def get_intersectional_group_data_subset(self, X, G, group_label, use_missing_val):
        q = ' & '.join(
            [f"""{n} == '{v}'""" for n, v in zip(G.columns, group_label) if v != MISSING_VALUE or use_missing_val])
        idx = G.query(q).index.tolist() if q else G.index.tolist()
        ig_names = [n for n, v in zip(G.columns, group_label) if (v != MISSING_VALUE or use_missing_val)]
        if set(self.required_group_attributes).issubset(ig_names) and (len(idx) > 0):
            G_ig = G.iloc[idx]
            to_drop = list(set(G_ig.columns) - set(self.required_group_attributes))
            G_ig = G_ig.drop(columns=to_drop)
            return X[idx], G_ig, idx
        else:
            return None, None, []

    def predict_for(self, X, G, group_label, use_missing_val=False):
        G = G.reset_index(drop=True)
        X_ig, G_ig, idx = self.get_intersectional_group_data_subset(X, G, group_label, use_missing_val)
        if not len(idx):
            return [], []
        return self.predict(X_ig, G_ig), idx

    def predict_proba_for(self, X, G, group_label, use_missing_val=False):
        X_ig, G_ig, idx = self.get_intersectional_group_data_subset(X, G, group_label, use_missing_val)
        if not len(idx):
            return [], []
        return self.predict_proba(X_ig, G_ig), idx
