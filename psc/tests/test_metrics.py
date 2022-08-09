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

import psc.metrics as metrics


@pytest.fixture
def y():
    return np.array([-1, -1, 1, 1])


@pytest.fixture
def pred_proba():
    return np.array([0.1, 0.4, 0.35, 0.8])


def test_compute_auc(y, pred_proba):
    assert metrics.compute_auc(y, pred_proba) == 0.75


def test_compute_log_loss(y, pred_proba):
    assert metrics.compute_log_loss(y, pred_proba) == pytest.approx(0.47228795380917615)


def test_compute_ece(y, pred_proba):
    assert metrics.compute_ece(y, pred_proba) == pytest.approx(0.1375)
