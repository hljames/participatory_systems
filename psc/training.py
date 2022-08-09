"""
Classes and functions to standardize the training of binary classification models
"""
import numpy as np
from inspect import signature

from psc.classification import LinearClassificationModel
from psc.groups import GroupAttributeEncoder

SUPPORTED_METHOD_NAMES = {'svm_linear', 'logreg'}

DEFAULT_TRAINING_SETTINGS = {

    'svm_linear': {
        'fit_intercept': True,
        'intercept_scaling': 1.0,
        'class_weight': None,
        'loss': "hinge",
        'penalty': 'l2',
        'C': 1.0,
        'tol': 1e-4,
        'max_iter': 1e3,
        'dual': True,
        'random_state': None,
        'verbose': False
    },

    'logreg': {
        'fit_intercept': True,
        'class_weight': None,
        'penalty': 'none',
        'tol': 1e-4,
        'solver': 'lbfgs',
        'warm_start': False,
        'max_iter': 1e5,
        'random_state': 2338,
        'verbose': True,
        'n_jobs': 1
    },
}


#### sklearn wrappers ####

def train_sklearn_linear_model(X, G, y, method_name, normalize_variables=False, settings=None):
    assert method_name in SUPPORTED_METHOD_NAMES, 'method %s not supported' % method_name
    # set missing settings
    if settings is None:
        settings = dict(DEFAULT_TRAINING_SETTINGS[method_name])
    else:
        assert isinstance(settings, dict)
        settings = dict(settings)
        for name, default_value in DEFAULT_TRAINING_SETTINGS[
            method_name].items():
            settings.setdefault(name, default_value)

    if G is not None:
        assert 'encoding_type' in settings, "encoding_type must be specified if G is not None"
        group_encoder = GroupAttributeEncoder(df=G, **settings)
        Z = group_encoder.to_dummies(df=G)
        if X is not None:
            X = np.hstack([Z, X])
        else:
            X = Z
    else:
        group_encoder = None
    # import correct classifier from scikit learn
    if method_name == 'svm_linear':
        from sklearn.svm import LinearSVC
        Classifier = LinearSVC
    elif method_name == 'logreg':
        from sklearn.linear_model import LogisticRegression
        Classifier = LogisticRegression

    # drop the intercept from the data if it exists
    coefficient_idx = range(X.shape[1])

    # preprocess features
    if normalize_variables:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(copy=True, with_mean=True,
                                with_std=True).fit(X)
        x_shift = np.array(scaler.mean_)
        x_scale = np.sqrt(scaler.var_)
        X = scaler.transform(X)
    else:
        x_shift = np.zeros(X.shape[1], dtype=float)
        x_scale = np.ones(X.shape[1], dtype=float)

    # extract classifier arguments from settings
    clf_args = dict()
    clf_argnames = list(signature(Classifier).parameters.keys())
    for k in clf_argnames:
        if k in settings and settings[k] is not None:
            clf_args[k] = settings[k]

    # fit classifier
    clf = Classifier(**clf_args)
    clf.fit(X, y)

    # store classifier parameters
    b = np.array(clf.intercept_) if settings['fit_intercept'] else 0.0
    w = np.array(clf.coef_)

    # adjust coefficients for unnormalized data
    if normalize_variables:
        w = w * x_scale
        b = b + np.dot(w, x_shift)

    w = np.array(w).flatten()
    b = float(b)

    # setup parameters for model object
    predict_handle = clf.predict
    proba_handle = clf.predict_proba

    model_info = {
        'intercept': b,
        'coefficients': w,
        'coefficient_idx': coefficient_idx,
        'impute_group_attributes': settings.get('impute_group_attributes', False)
    }

    training_info = {
        'method_name': method_name,
        'normalize_variables': normalize_variables,
        'x_shift': x_shift,
        'x_scale': x_scale,
        'n_training_samples': len(y)
    }

    training_info.update(settings)
    model = LinearClassificationModel(predict_handle=predict_handle,
                                      proba_handle=proba_handle,
                                      group_encoder=group_encoder,
                                      model_type=LinearClassificationModel.LINEAR_MODEL_TYPE,
                                      model_info=model_info,
                                      training_info=training_info)

    return model


settings = {
    # dataset parameters
    'data_name': 'heart',
    'rebalancing_type': 'yg',
    # equalize positive and negative samples for each group, then equalize number of samples per group
    'fold_id': 'K05N01',  # five folds, first iteration
    'fold_num_validation': 4,
    'fold_num_test': 5,
    # method parameters
    'method_name': 'logreg',
    'encoding_type': 'intersectional',
    'seed': 2337,
    'normalize_variables': False
}
