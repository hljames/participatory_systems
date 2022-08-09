"""
Install pytest and pytest-cov:
pip3 install pytest
pip3 install pytest-cov

To run all tests:
pytest

To run all tests and compute test coverage:
pytest --cov
"""

import numpy as np
import pytest

from psc import metrics
from psc.auditor import PersonalizationAuditor
from psc.data import BinaryClassificationDataset
from psc.participatory import FlatSystem, SequentialSystem
from psc.paths import get_processed_data_file
from psc.training import train_sklearn_linear_model
from psc.utils import powerset


@pytest.fixture
def settings():
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
    }
    return settings


@pytest.fixture
def data(settings):
    data_file_processed = get_processed_data_file(settings['data_name'], rebalancing_type=settings['rebalancing_type'])
    data = BinaryClassificationDataset.load(file=data_file_processed)
    data.split(fold_id=settings['fold_id'], fold_num_validation=settings['fold_num_validation'],
               fold_num_test=settings['fold_num_test'])
    return data


@pytest.fixture
def generic_model(settings, data):
    if settings['method_name'] in ('logreg'):
        train_model = lambda X, G, y, settings, normalize_variables: train_sklearn_linear_model(X, G, y,
                                                                                                method_name=settings[
                                                                                                    'method_name'],
                                                                                                settings=settings,
                                                                                                normalize_variables=normalize_variables)
    else:
        raise NotImplementedError()
    generic_model = train_model(X=data.training.X, G=None, y=data.training.y, settings=settings,
                                normalize_variables=False)
    return generic_model


@pytest.fixture
def personalized_model(settings, data):
    if settings['method_name'] in ('logreg'):
        train_model = lambda X, G, y, settings, normalize_variables: train_sklearn_linear_model(X, G, y,
                                                                                                method_name=settings[
                                                                                                    'method_name'],
                                                                                                settings=settings,
                                                                                                normalize_variables=normalize_variables)
    else:
        raise NotImplementedError()
    personalized_model = train_model(X=data.training.X, G=data.training.G, y=data.training.y, settings=settings,
                                     normalize_variables=False)
    return personalized_model


@pytest.fixture
def auditor(data, generic_model):
    auditor = PersonalizationAuditor(data=data, generic_model=generic_model)
    return auditor


@pytest.fixture
def candidate_models(data, settings):
    candidate_models = []
    for g_names in powerset(data.group_attributes.names, min_size=1):
        G = data.training.G[list(g_names)]
        h = train_sklearn_linear_model(data.training.X, G, data.training.y,
                                       method_name=settings['method_name'],
                                       settings=settings,
                                       normalize_variables=False)
        candidate_models.append(h)
    return candidate_models


def test_compute_performance_metrics_error(data, generic_model, candidate_models):
    auditor = PersonalizationAuditor(data=data, generic_model=generic_model, assignment_metric='error',
                                     sample_type='training')
    generic_train_perf = auditor.compute_performance_metrics(model=generic_model,
                                                             model_type='generic',
                                                             metric_name='error',
                                                             sample_type='training')
    p_seq = SequentialSystem(data, generic_model, assignment_metric='error', assignment_sample='training')
    p_seq.update_assignments(candidate_models)
    p_seq_train_perf = auditor.compute_performance_metrics(model=p_seq,
                                                           model_type='personalized',
                                                           metric_name='error',
                                                           sample_type='training')
    p_flat = FlatSystem(data, generic_model, assignment_metric='error', assignment_sample='training')
    p_flat.update_assignments(candidate_models)
    p_flat_train_perf = auditor.compute_performance_metrics(model=p_flat,
                                                            model_type='personalized',
                                                            metric_name='error',
                                                            sample_type='training')
    assert all(np.asarray(generic_train_perf) >= np.asarray(p_seq_train_perf))
    assert all(np.asarray(generic_train_perf) >= np.asarray(p_flat_train_perf))
    assert list(p_seq_train_perf) == list(p_flat_train_perf)


def test_compute_performance_metrics_ece(data, generic_model, auditor, candidate_models):
    generic_train_perf = auditor.compute_performance_metrics(model=generic_model,
                                                             model_type='generic',
                                                             metric_name='ece',
                                                             sample_type='training')
    p_seq = SequentialSystem(data, generic_model, assignment_metric='ece', assignment_sample='training')
    p_seq.update_assignments(candidate_models)
    p_seq_train_perf = auditor.compute_performance_metrics(model=p_seq,
                                                           model_type='personalized',
                                                           metric_name='ece',
                                                           sample_type='training')
    p_flat = FlatSystem(data, generic_model, assignment_metric='ece', assignment_sample='training')
    p_flat.update_assignments(candidate_models)
    p_flat_train_perf = auditor.compute_performance_metrics(model=p_flat,
                                                            model_type='personalized',
                                                            metric_name='ece',
                                                            sample_type='training')
    assert all(np.asarray(generic_train_perf) >= np.asarray(p_seq_train_perf))
    assert all(np.asarray(generic_train_perf) >= np.asarray(p_flat_train_perf))
    assert list(p_seq_train_perf) == list(p_flat_train_perf)


def test_compute_performance_metrics_auc(data, generic_model, auditor, candidate_models):
    generic_train_perf = auditor.compute_performance_metrics(model=generic_model,
                                                             model_type='generic',
                                                             metric_name='auc',
                                                             sample_type='training')
    p_seq = SequentialSystem(data, generic_model, assignment_metric='auc', sample_type='training')
    p_seq.update_assignments(candidate_models)
    p_seq_train_perf = auditor.compute_performance_metrics(model=p_seq,
                                                           model_type='personalized',
                                                           metric_name='auc',
                                                           sample_type='training')
    p_flat = FlatSystem(data, generic_model, assignment_metric='auc', sample_type='training')
    p_flat.update_assignments(candidate_models)
    p_flat_train_perf = auditor.compute_performance_metrics(model=p_flat,
                                                            model_type='personalized',
                                                            metric_name='auc',
                                                            sample_type='training')
    assert all(np.asarray(generic_train_perf) <= np.asarray(p_seq_train_perf))
    assert all(np.asarray(generic_train_perf) <= np.asarray(p_flat_train_perf))
    assert list(p_seq_train_perf) == list(p_flat_train_perf)


def test_compute_performance_metrics_log_loss(data, generic_model, personalized_model, auditor, candidate_models):
    generic_train_perf = auditor.compute_performance_metrics(model=generic_model,
                                                             model_type='generic',
                                                             metric_name='log_loss',
                                                             sample_type='training')
    personalized_train_perf = auditor.compute_performance_metrics(model=personalized_model,
                                                                  model_type='personalized',
                                                                  metric_name='log_loss',
                                                                  sample_type='training')
    p_seq = SequentialSystem(data, generic_model, assignment_metric='log_loss', assignment_sample='training')
    p_seq.update_assignments(candidate_models)
    p_seq_train_perf = auditor.compute_performance_metrics(model=p_seq,
                                                           model_type='personalized',
                                                           metric_name='log_loss',
                                                           sample_type='training')
    p_flat = FlatSystem(data, generic_model, assignment_metric='log_loss', assignment_sample='training')
    p_flat.update_assignments(candidate_models)
    p_flat_train_perf = auditor.compute_performance_metrics(model=p_flat,
                                                            model_type='personalized',
                                                            metric_name='log_loss',
                                                            sample_type='training')
    # Loss should be GREATER for generic/personalized compared with flat/sequential
    assert all(np.asarray(generic_train_perf) >= np.asarray(p_flat_train_perf))
    assert all(np.asarray(generic_train_perf) >= np.asarray(p_seq_train_perf))
    assert all(np.asarray(personalized_train_perf) >= np.asarray(p_seq_train_perf))
    assert all(np.asarray(personalized_train_perf) >= np.asarray(p_flat_train_perf))
    assert list(p_seq_train_perf) == list(p_flat_train_perf)


def test_generate_bootstrap_indices(data, auditor):
    yG = np.hstack((data.training.y[:, np.newaxis], np.array(data.training.G.values, dtype=str)))
    _, yg_prev = np.unique(yG, axis=0, return_index=False, return_counts=True)
    yg_prev = yg_prev / len(yG)
    training_bootstrap_inds = auditor.generate_bootstrap_indices(data.training.y, data.training.G)
    for ib in training_bootstrap_inds:
        _, yg_prev_ib = np.unique(yG[ib], axis=0, return_index=False, return_counts=True)
        yg_prev_ib = yg_prev_ib / len(ib)
        np.testing.assert_array_almost_equal(yg_prev_ib, yg_prev)
