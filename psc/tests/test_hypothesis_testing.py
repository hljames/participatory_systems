"""
Helper functions to run hypothesis tests
"""
import numpy as np
from scipy import stats as st
from sklearn.utils import resample


def exact_mcnemar_test(y, yhat1, yhat2, two_sided=False):
    """
    :param y: true labels
    :param yhat1: predictions of classifier 1
    :param yhat2: predictions of classifier 2
    :param two_sided: set to True for two-sided test
    :return: value of the McNemar Test
    """

    f1_correct = np.equal(y, yhat1)
    f2_correct = np.equal(y, yhat2)

    table = np.array([
        [np.sum((f1_correct == 0) & (f2_correct == 0)),
         np.sum((f1_correct == 0) & (f2_correct == 1))],
        [np.sum((f1_correct == 1) & (f2_correct == 0)),
         np.sum((f1_correct == 1) & (f2_correct == 1))]
    ])

    # table = np.zeros(shape = (2, 2))
    #   for i in range(2):
    #       for j in range(2):
    #           table[i, j] = np.sum((f1_correct == i) & (f2_correct == j))

    b = table[0, 1]  # f1 wrong and f2 right
    c = table[1, 0]  # f1 right and f2 wrong
    n = b + c

    # envy-freeness requires that
    # f1 is correct more often than f2 <=> b < c
    #
    # We test
    #
    # H0: error(f1) = error(f2)
    # H1: error(f1) > error(f2)
    #
    # This requires assuming b/(b+c) ~ Bin(0.5)

    if not two_sided:
        test_statistic = c
        p = st.binom.cdf(k=test_statistic, n=n, p=0.5)
    else:
        test_statistic = min(b, c)
        p = 2.0 * st.binom.cdf(k=min(b, c), n=b + c, p=0.5)

    return p, test_statistic


##### Bootstrap Testing
def generate_bootstrap_indices(ys, group_values, stratify=True,
                               random_seed=1337):
    """
    :param data:
    :param n_samples:
    :return:
    """
    group_values_with_label = np.concatenate((group_values, ys[:, np.newaxis]),
                                             axis=1)

    # get unique ids for each combination of group attributes
    _, index, profile_idx, profile_counts = np.unique(group_values_with_label,
                                                      axis=0,
                                                      return_index=True,
                                                      return_inverse=True,
                                                      return_counts=True)
    random_state = np.random.RandomState(random_seed)
    bootstrap_sample = [
        resample(np.array(range(len(ys))), random_state=random_state,
                 stratify=profile_idx) for i in range(1000)]
    return np.array(bootstrap_sample)


def two_sample_paired_sign_test(S1, S2, alternative='greater'):
    """
    run a two-sample paired sign test to compare performance of classifier 1 vs. classifier 2

        H0: performance(h1) = performance(h2)
        HA: performance(h1) < performance(h2)

    :param S1: vector of m values of the performance metric - each value is the performance of classifier 1 on m datasets
    :param S2: vector of m values of the performance metric - each value is the performance of classifier 2 on m datasets

    :param alternative: sign of HA. must be 'greater' (default) or 'less'
           alternative = 'greater' -> H0: performance(h1) > performance(h2)   HA: performance(h1) < performance(h2)
           alternative = 'less'   ->  H0: performance(h1) < performance(h2)   HA: performance(h1) > performance(h2)

    :return: p-value, test_stat where
             p_value = P(observe performance of S1 and S2 | H0)
             test_stat = # times where HA was true

    ----------------------------------------------------------------------------
    Usage
    ----------------------------------------------------------------------------

    Use this function to compare the performance of two classifiers via tests of the form

    The tests have the form:

      H0: R(h1, D) < R(h2, D)
      HA: R(h1, D) > R(h2, D)

    where:

    - R(clf, D) is a function to compute the performance of a model clf on data D
    - D is a sample of (G, X, y)

    ----------------------------------------------------------------------------
    Example: To "test rationality of 'group k' for 'ece' on 'validation' data

    We let

      h1 = generic model
      h2 = personalized model
      D = validation dataset

    Thus, we have that:

      H0: R(generic_model, data_validation) < R(personalized_model, data_validation)

    And the p-value of this test represents:

      Pr(observe S1 and S2 | no fair use violation)

    We can run this test with this function as follows:

    # 1: Construct 1000 copies of the validation dataset for group k via bootstrap resampling:

      Ik = np.all(data.validation.G.values == ('Female', 'Old'), axis = 1)
      Gk = data.validation.G[Ik]
      Xk = data.validation.X[Ik]
      yk = data.validation.y[Ik]

      I_bootstrap = self.bootstrap_indices['validation'][Ik, :]
      D_bootstrap = [(Gk[idx], X[idx], y[idx])) for idx in I_bootstrap]

    # 2: Compute the expected_calibration error of h1 and h2 for each bootstrapped dataset

      S1 = [get_ece(h1, Xb, yb) for (Gb, Xb, yb) in D_bootstrap]
      S2 = [get_ece(h2, Xb, yb) for (Gb, Xb, yb) in D_bootstrap]

    # 3: Call this function

      pvalue, test_stat = two_sample_paired_sign_test(S1, S2, alternative = 'greater')

    """
    S1 = np.array(S1)
    S2 = np.array(S2)
    m = len(S1)
    assert m > 0
    assert len(S2) == m
    assert alternative in ('greater', 'less')
    test_stat = np.greater_equal(S1 - S2, 0.0).sum()
    p_value = st.binom_test(test_stat, n=len(S1), p=0.5,
                            alternative=alternative)

    return p_value, test_stat
