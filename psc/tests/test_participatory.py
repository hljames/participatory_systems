"""
Install pytest and pytest-cov:
pip3 install pytest
pip3 install pytest-cov

To run all tests:
pytest

To run all tests and compute test coverage:
pytest --cov
"""

import pytest

from psc import metrics
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


def test_generate_all_trees(data, generic_model, settings):
    p_seq = SequentialSystem(data, generic_model)
    p_seq.label_map = {'Sex': ['Female', 'Male'], 'Age': ['Old', 'Young'], 'Blood_Type': ['+', '-']}
    p_seq.generate_all_trees()
    assert len(p_seq.all_trees) == 24
    # assert all trees are unique
    assert len(set(p_seq.all_trees)) == 24

    # # time intensive
    # p_seq.label_map = {'Sex': ['Female', 'Male'], 'Age': ['Old', 'Young'], 'Blood_Type': ['A+', 'A-', 'B+', 'B-',
    #                                                                                       'O+', 'O-', 'AB+', 'AB-']}
    # p_seq.generate_all_trees()
    # assert len(p_seq.all_trees) == 528
    # assert len(set(p_seq.all_trees)) == 528
