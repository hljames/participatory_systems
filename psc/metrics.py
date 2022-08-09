"""
Helper functions to compute standard classification metrics
"""
import numpy as np

predict_handles_map = {
    'auc': 'predict_proba',
    'ece': 'predict_proba',
    'log_loss': 'predict_proba',
    'error': 'predict',
    'n_mistakes': 'predict'
}


def compute_error(y_true, y_pred):
    """
    computes error rate for a binary classifier
    :param y_true:
    :param y_pred: predictions
    :return:
    """
    return np.not_equal(y_true, y_pred).mean()


def compute_auc(y_true, y_proba):
    """
    computes AUC for a binary classifier quickly
    :param y_true: vector of true classes
    :param y_proba: vector of predicted probabilities
    :return: auc
    """
    n = len(y_true)
    I_pos = y_true > 0
    # if y[i] == 0 for all i or y[i] == 1 for all i, then return AUC = 1
    if I_pos.all() or np.logical_not(I_pos).all():
        return float('nan')

    I_pos = I_pos[np.argsort(y_proba)]
    false_positive_count = np.cumsum(1 - I_pos)
    n_false = false_positive_count[-1]
    auc = np.multiply(I_pos, false_positive_count).sum() / (
            n_false * (n - n_false))
    return auc


def compute_n_mistakes(y_true, y_pred):
    """
    computes AUC for a binary classifier quickly
    :param y_true: vector of true classes
    :param y_pred: vector of predicted probabilities
    :return: auc
    """
    return str(sum(y_true != y_pred)) + '/' + str(len(y_true))


def compute_ece(y_true, y_proba, n_bins=10):
    """
    computes the expected calibration error quickly
    :param y_true: vector of true labels
    :param y_proba: vector of predicted probabilities
    :param n_bins: 10
    :return: expected calibration error (L1)
    """
    # pre-process to improve binning
    sort_idx = np.argsort(y_proba)
    ys_prob = y_proba[sort_idx]
    ys_diff = ys_prob - (y_true[sort_idx] > 0)
    err = 0.0
    i = 0
    for k in range(1, n_bins + 1):
        j = np.searchsorted(ys_prob, k / n_bins, side='right')
        if i < j:
            err += np.abs(np.sum(ys_diff[i:j]))
        i = j
    ece = err / len(y_true)
    return ece


def compute_log_loss(y_true, y_proba, eps=1e-15):
    """
    computes mean logistic loss using labels and probability predictions
    :param y_true: vector of true labels - y_true[i] in (-1,+1) and y_true[i] (0,1) are both OK
    :param y_proba: vector of predicted probabilities
    :param eps: minimum distance that y_pred must maintain from 0 and 1
    :return: normalized logistic loss where:

              y = y_true > 0
              L = y * np.log(p) + (1 - y) * np.log(1.0 - p)
              log_loss = np.mean(L)

    -----------------------
    Note that the implemntation is meant to be fast, but isn't easy to understand.

    Here are two other implementations to clarify what's going on

    ##### basic #####

    p = np.clip(y_pred, eps, 1.0 - eps)
    y = y_true > 0
    L = y * np.log(p) + (1 - y) * np.log(1.0 - p)
    return np.mean(L)

    ##### faster (only calls log once) #####

    p = np.clip(y_pred, eps, 1.0 - eps) # clip probabilities
    I_pos = y_true > 0                  # convert y_true to y in (0,1) /indices of true points
    L = np.empty_like(y_true)  
    L[idx_pos] = p[I_pos]
    L[~idx_pos] = 1 - p[~I_pos]
    return np.log(L).mean()

    """
    #### fastest implementation (only calls log once / minimizes storage)

    # L will eventually contain the values of the log-loss at each point

    # Initialize L = p
    L = np.clip(y_proba, eps, 1.0 - eps)

    # Let I_neg = i where y_true[i] == 0
    I_neg = y_true < 1

    # Set L[i] = 1 - p[i] for all i where y_true[i] ≠ 1
    # L[I_neg] -> -L[I_neg] -> 1 - L[I_neg]
    L[I_neg] *= -1
    L[I_neg] += 1

    return -1 * np.log(L).mean()


##### older implementations (use for testing)

def log_loss_basic(y_true, y_proba, eps=1e-15):
    """
    computes mean logistic loss using labels and probability predictions
    :param y_true: vector of true labels - y_true[i] in (-1,+1) and y_true[i] (0,1) are both OK
    :param y_proba: vector of predicted probabilities
    :param eps: minimum distance that y_pred must maintain from 0 and 1
    :return: log_loss
    """
    p = np.clip(y_proba, eps, 1.0 - eps)
    y = y_true > 0
    L = y * np.log(p) + (1 - y) * np.log(1.0 - p)
    return np.mean(L)


def log_loss_fast(y_true, y_proba, eps=1e-15):
    """
    computes mean logistic loss using labels and probability predictions
    :param y_true: vector of true labels - y_true[i] in (-1,+1) and y_true[i] (0,1) are both OK
    :param y_proba: vector of predicted probabilities
    :param eps: minimum distance that y_pred must maintain from 0 and 1
    :return: log_loss
    """
    p = np.clip(y_proba, eps, 1.0 - eps)  # clip probabilities
    I_pos = y_true > 0  # convert y_true to y in (0,1) /indices of true points
    L = np.empty_like(y_true)
    L[I_pos] = p[I_pos]
    L[~I_pos] = 1 - p[~I_pos]
    return np.log(L).mean()


def ece_score_basic(y_true, y_proba, n_bins=10):
    """
    computes expected calibration error for a binary classifier
    :param y_true: vector of true classes
    :param y_proba: vector of predicted probabilities
    :param n_bins: 10
    :return:ece
    """

    # pre-process to improve binning
    sort_idx = np.argsort(y_proba)
    y_pred = y_proba[sort_idx]
    y_true = y_true[sort_idx] > 0
    n = len(y_true)

    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for k in range(n_bins):
        left_idx = np.searchsorted(y_pred, k / n_bins, side='left')
        right_idx = np.searchsorted(y_pred, (k + 1) / n_bins, side='right')
        if left_idx < right_idx:
            bin_idx = np.arange(left_idx, right_idx)
            counts[k] = len(bin_idx)
            conf[k] = np.mean(y_pred[bin_idx])
            acc[k] = np.mean(y_true[bin_idx])

    ece = np.sum(counts * np.abs(acc - conf)) / n
    return ece


def ece_score_fast(y_true, y_proba, n_bins=10):
    """
    computes expected calibration error for a binary classifier (quicker than basic)
    :param y_true: vector of true classes
    :param y_proba: vector of predicted probabilities
    :param n_bins: 10
    :return:ece
    """

    sort_idx = np.argsort(y_proba)
    ys_prob = y_proba[sort_idx]
    ys_true = y_true[sort_idx] > 0
    counts, conf, acc = np.zeros(n_bins, dtype=np.int_), np.zeros(
        n_bins), np.zeros(n_bins)
    a = 0
    for k in range(n_bins):
        # a = np.searchsorted(ys_prob, k / n_bins, side = 'left')
        b = np.searchsorted(ys_prob, (k + 1) / n_bins, side='right')
        counts[k] = b - a
        if counts[k] > 0:
            conf[k] = np.mean(ys_prob[a:b])
            acc[k] = np.mean(ys_true[a:b])
        a = b
    ece = np.sum(counts * np.abs(acc - conf)) / len(y_true)
    return ece
