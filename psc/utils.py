"""
This file contains general purpose helper functions
"""
import numpy as np
import re
import time
import sys
from itertools import chain, combinations
from copy import deepcopy

from psc.groups import MISSING_VALUE

METRIC_NAMES = ['log_loss', 'auc', 'ece', 'error']

PARTICIPATORY_MODEL_TYPES = ['participatory_simple', 'sequential', 'flat']

PARTIAL_REPORTING_MODEL_TYPES = ['sequential', 'flat']

MODEL_TYPES = PARTICIPATORY_MODEL_TYPES + ['generic', 'personalized', 'onehot_impute', 'onehot', 'intersectional',
                                           'decoupled']


def raw_gain(p1, p2, metric_name):
    assert metric_name in METRIC_NAMES
    if metric_name in ['auc']:
        return p2 - p1
    elif metric_name in ['error', 'ece', 'log_loss']:
        return p1 - p2


#### printing
_LOG_TIME_FORMAT = "%m/%d/%y @ %I:%M %p"


def print_log(msg, print_flag=True):
    """this function replaces print"""
    if print_flag:
        if isinstance(msg, str):
            print_str = '%s | %s' % (time.strftime(_LOG_TIME_FORMAT, time.localtime()), msg)
        else:
            print_str = '%s | %r' % (time.strftime(_LOG_TIME_FORMAT, time.localtime()), msg)
        print(print_str)
        sys.stdout.flush()


#### iterators
def powerset(iterable, min_size=0):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    xs = list(iterable)
    # note we return an iterator rather than a list

    return chain.from_iterable(
        combinations(xs, n) for n in range(min_size, len(xs) + 1))


def product(*args, repeat=1):
    """
    Cartesian product with deep copies
    EX1: product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    EX2: product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    :param args: lists for cartesian product
    :param repeat: number of times to repeat
    :return: iterable
    """
    pools = [tuple(deepcopy(pool)) for pool in args] * repeat
    result = [[]]
    for pool in pools:
        result = [deepcopy(x) + [deepcopy(y)] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def clean_latex_str(s):
    s = re.sub(r"(?<!\\)_", "\_", s)
    s = s.replace(MISSING_VALUE, 'NA')
    s = s.lstrip('\n')
    s = s.rstrip('\n')
    return s


def metric_latex(value, metric_name):
    assert metric_name in METRIC_NAMES
    if metric_name in ('auc', 'log_loss'):
        return f'{value:.3f}'
    else:
        return f'{value * 100:.1f}\\%'


def group_name_latex(ig):
    if isinstance(ig, tuple):
        return '\\ \\&\\ '.join(ig).lower().replace(MISSING_VALUE, 'NA')
    elif isinstance(ig, str):
        return ig.replace(MISSING_VALUE, 'NA')
    else:
        return ig
